from dotenv import load_dotenv
import logging, os, sys, time

from airport_data.airport_data import AirportData
from flight_input_parser.flight_input_parser import FlightInputParser
from airport_code_validator.airport_code_validator import AirportCodeValidator
from flight_path.flight_path import FlightPath
from notam_fetcher import NotamFetcher
from notam_fetcher.api_schema import CoreNOTAMData, Notam
from notam_fetcher.exceptions import NotamFetcherRateLimitError, NotamFetcherRequestError, NotamFetcherUnauthenticatedError
from notam_printer.notam_printer import NotamPrinter
from sorting_algorithm.sorting_algorithm import NotamSorter

# ---------------------------------------------------------------------------
# ML integration — graceful degradation if models not yet trained
# ---------------------------------------------------------------------------
try:
    from ml.ml_scorer import MLScorer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


log_format_string = '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
log_formatter = logging.Formatter(log_format_string)
logging.basicConfig(level=logging.DEBUG, format=log_format_string, force=True)

# Also log to a file
log_file_handler = logging.FileHandler('output.log', mode='w')
log_file_handler.setLevel(logging.DEBUG)
log_file_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_file_handler)

# Silence chatty HTTP debug loggers
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

logger = logging.getLogger("driver")


def main():
    """
    Main execution block:
    - Load environment variables for CLIENT_ID and CLIENT_SECRET
    - Validates airport codes and computes flight path.
    - Fetches NOTAMs along the route.
    - Scores NOTAMs with ML pipeline (criticality + anomaly detection).
    - Saves ML-sorted brief to data/raw/<DEP>_<DEST>_ml.txt
    - Saves raw NOTAM list to data/raw/<DEP>_<DEST>.txt
    """

    load_dotenv()
    CLIENT_ID = os.getenv("CLIENT_ID")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")

    if CLIENT_ID is None:
        logger.error("CLIENT_ID not set in .env file")
        sys.exit("Error: CLIENT_ID not set in .env file")
    if CLIENT_SECRET is None:
        sys.exit("Error: CLIENT_SECRET not set in .env file")

    departure_airport_code, destination_airport_code = FlightInputParser.get_flight_input()

    try:
        departure_airport   = AirportData.get_airport(departure_airport_code)
        destination_airport = AirportData.get_airport(destination_airport_code)
    except ValueError as e:
        sys.exit(str(e))

    if not AirportCodeValidator.is_valid(departure_airport):
        sys.exit(f"Invalid departure airport {departure_airport}. Please enter valid airport codes.")
    if not AirportCodeValidator.is_valid(destination_airport):
        sys.exit(f"Invalid destination airport {destination_airport}. Please enter valid airport codes.")

    logger.info(f"Fetching Flights from {departure_airport.icao} to {destination_airport.icao}")

    flight_path = FlightPath(departure_airport, destination_airport)
    waypoints   = flight_path.get_waypoints_by_gap(40)

    notam_fetcher = NotamFetcher(CLIENT_ID, CLIENT_SECRET, timeout=300)
    all_notams: list[CoreNOTAMData] = []

    start_time = time.perf_counter()
    try:
        all_notams = notam_fetcher.fetch_notams_by_latlong_list(waypoints, 30)
    except NotamFetcherUnauthenticatedError:
        logging.error("Invalid client_id or secret.")
        sys.exit("Invalid client_id or secret.")
    except NotamFetcherRequestError:
        logging.error("Failed to retrieve NOTAMs due to a network issue.")
        sys.exit("Failed to retrieve NOTAMs due to a network issue.")
    except NotamFetcherRateLimitError:
        logging.error("Failed to retrieve NOTAMs due to rate limits.")
        sys.exit("Failed to retrieve NOTAMs due to rate limits.")
    end_time = time.perf_counter()

    notams = [notam.notam for notam in all_notams]
    logger.info(f"Fetched {len(notams)} unique NOTAMs in {end_time - start_time:.3f} seconds")

    # ------------------------------------------------------------------
    # Step 1: Deduplicate via NotamSorter
    # ------------------------------------------------------------------
    sorter        = NotamSorter(notams)
    sorted_notams = sorter.sort_by_score()

    # ------------------------------------------------------------------
    # Step 2: ML scoring — save sorted brief to file
    # ------------------------------------------------------------------
    if ML_AVAILABLE:
        try:
            ml_scorer  = MLScorer()
            ml_results = ml_scorer.score(sorted_notams)
            ml_brief   = f"data/raw/{departure_airport_code}_{destination_airport_code}_ml.txt"
            ml_scorer.save_brief(
                ml_results,
                filepath=ml_brief,
                departure=departure_airport_code,
                destination=destination_airport_code,
            )
            logger.info(f"ML brief saved to {ml_brief}")
        except FileNotFoundError as e:
            logger.warning(f"ML models not found — skipping ML scoring. ({e})")
            logger.warning("Run notebooks/03_model_training.ipynb to train models.")
        except Exception as e:
            logger.error(f"ML scoring failed unexpectedly: {e}")
    else:
        logger.info("ML module not available.")

    # ------------------------------------------------------------------
    # Step 3: Save raw NOTAM list to file (original format)
    # ------------------------------------------------------------------
    printer     = NotamPrinter()
    output_file = printer.save_to_file(
        sorted_notams,
        filepath=f"data/raw/{departure_airport_code}_{destination_airport_code}.txt"
    )
    logger.info(f"Saved {len(sorted_notams)} NOTAMs to {output_file}")


if __name__ == "__main__":
    main()