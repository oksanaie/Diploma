DATE_TIME=0 # Timestamp
SITE_NAME=1 # ID of the Expedia point of sale (i.e. Expedia.com, Expedia.co.uk, 
            # Expedia.co.jp, ...) 
POSA_CONTINENT=2 # ID of continent associated with site_name
USER_LOCATION_COUNTRY=3 # The ID of the country the customer is located 
USER_LOCATION_REGION=4 # The ID of the region the customer is located
USER_LOCATION_CITY=5 # The ID of the city the customer is located
ORIG_DESTINATION_DISTANCE=6 # Physical distance between a hotel and a customer 
                            # at the time of search. A null means the distance could 
                            # not be calculated
USER_ID=7 # ID of user
IS_MOBILE=8 # 1 when a user connected from a mobile device, 0 otherwise
IS_PACKAGE=9 # 1 if the click/booking was generated as a part of a package 
             # (i.e. combined with a flight), 0 otherwise
CHANNEL=10 # ID of a marketing channel
SRCH_CI=11 # Checkin date
SRCH_CO=12 # Checkout date
SRCH_ADULTS_CNT=13 # The number of adults specified in the hotel room   
SRCH_CHILDREN_CNT=14 # The number of (extra occupancy) children specified in the 
                     # hotel room
SRCH_RM_CNT=15 # The number of hotel rooms specified in the search
SRCH_DESTINATION_ID=16 # ID of the destination where the hotel search was performed
SRCH_DESTINATION_TYPE_ID=17 # Type of destination
HOTEL_CONTINENT=18 # Hotel continent
HOTEL_COUNTRY=19 # Hotel country
HOTEL_MARKET=20 # Hotel market

# After all cutting is done for the train or test data we expect
# following features to remain
EXPECTED_HEADER=[
	"date_time", 
	"site_name", 
	"posa_continent", 
	"user_location_country", 
	"user_location_region",
	"user_location_city",
	"orig_destination_distance",
	"user_id",
	"is_mobile",
	"is_package",
	"channel",
	"srch_ci",
	"srch_co",
	"srch_adults_cnt",
	"srch_children_cnt",
	"srch_rm_cnt",
	"srch_destination_id",
	"srch_destination_type_id",
	"hotel_continent",
	"hotel_country",
	"hotel_market"]