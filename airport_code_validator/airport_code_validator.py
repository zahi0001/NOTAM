
from airport_data.types import Airport

EXCLUDED_TIMEZONES = [
    "America/Anchorage", "America/Juneau", "America/Sitka",
    "America/Metlakatla", "America/Yakutat", "America/Nome",
    "America/Adak", "Pacific/Honolulu"
]

EXCLUDED_REGION_CODES = [
    "America/Puerto_Rico", "Pacific/Guam", "Pacific/Pago_Pago",
    "Pacific/Saipan"
]


'''
Airport Code Validator Component

Features:
    - Validates if airport code exists and is from the Continental United States
'''
class AirportCodeValidator: 
    @staticmethod
    def is_valid(airport: Airport):
        """
        Validates if the airport is part of Continental United States

        Args:
            airport (Airport): Airport object.

        Returns:
            True: Valid Continental United States airport
            False: Airport outside Continental United States.
        """
        if airport.country != "United States":
            return False
        if airport.tz_name in EXCLUDED_TIMEZONES + EXCLUDED_REGION_CODES:
            return False
        return True