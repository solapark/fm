import enum
property_list = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star', 'Accessible Hotel', 'Accessible Parking', 'Adults Only', 'Air Conditioning', 'Airport Hotel', 'Airport Shuttle', 'All Inclusive (Upon Inquiry)', 'Balcony', 'Bathtub', 'Beach', 'Beach Bar', 'Beauty Salon', 'Bed & Breakfast', 'Bike Rental', 'Boat Rental', 'Body Treatments', 'Boutique Hotel', 'Bowling', 'Bungalows', 'Business Centre', 'Business Hotel', 'Cable TV', 'Camping Site', 'Car Park', 'Casa Rural (ES)', 'Casino (Hotel)', 'Central Heating', 'Childcare', 'Club Hotel', 'Computer with Internet', 'Concierge', 'Conference Rooms', 'Convenience Store', 'Convention Hotel', 'Cosmetic Mirror', 'Cot', 'Country Hotel', 'Deck Chairs', 'Design Hotel', 'Desk', 'Direct beach access', 'Diving', 'Doctor On-Site', 'Eco-Friendly hotel', 'Electric Kettle', 'Excellent Rating', 'Express Check-In / Check-Out', 'Family Friendly', 'Fan', 'Farmstay', 'Fitness', 'Flatscreen TV', 'Free WiFi (Combined)', 'Free WiFi (Public Areas)', 'Free WiFi (Rooms)', 'Fridge', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars', 'Gay-friendly', 'Golf Course', 'Good Rating', 'Guest House', 'Gym', 'Hairdresser', 'Hairdryer', 'Halal Food', 'Hammam', 'Health Retreat', 'Hiking Trail', 'Honeymoon', 'Horse Riding', 'Hostal (ES)', 'Hostel', 'Hot Stone Massage', 'Hotel', 'Hotel Bar', 'House / Apartment', 'Hydrotherapy', 'Hypoallergenic Bedding', 'Hypoallergenic Rooms', 'Ironing Board', 'Jacuzzi (Hotel)', "Kids' Club", 'Kosher Food', 'Large Groups', 'Laundry Service', 'Lift', 'Luxury Hotel', 'Massage', 'Microwave', 'Minigolf', 'Motel', 'Nightclub', 'Non-Smoking Rooms', 'On-Site Boutique Shopping', 'Openable Windows', 'Organised Activities', 'Pet Friendly', 'Playground', 'Pool Table', 'Porter', 'Pousada (BR)', 'Radio', 'Reception (24/7)', 'Resort', 'Restaurant', 'Romantic', 'Room Service', 'Room Service (24/7)', 'Safe (Hotel)', 'Safe (Rooms)', 'Sailing', 'Satellite TV', 'Satisfactory Rating', 'Sauna', 'Self Catering', 'Senior Travellers', 'Serviced Apartment', 'Shooting Sports', 'Shower', 'Singles', 'Sitting Area (Rooms)', 'Ski Resort', 'Skiing', 'Solarium', 'Spa (Wellness Facility)', 'Spa Hotel', 'Steam Room', 'Sun Umbrellas', 'Surfing', 'Swimming Pool (Bar)', 'Swimming Pool (Combined Filter)', 'Swimming Pool (Indoor)', 'Swimming Pool (Outdoor)', 'Szep Kartya', 'Table Tennis', 'Telephone', 'Teleprinter', 'Television', 'Tennis Court', 'Tennis Court (Indoor)', 'Terrace (Hotel)', 'Theme Hotel', 'Towels', 'Very Good Rating', 'Volleyball', 'Washing Machine', 'Water Slide', 'Wheelchair Accessible', 'WiFi (Public Areas)', 'WiFi (Rooms)']
filter_list = ['1 Night', '1 Star', '2 Nights', '2 Star', '3 Nights', '3 Star', '4 Star', '5 Nights', '5 Star', 'Accessible Hotel', 'Accessible Parking', 'Adults Only', 'Air Conditioning', 'Airport Hotel', 'Airport Shuttle', 'All Inclusive (Upon Inquiry)', 'Balcony', 'Bathtub', 'Beach', 'Beach Bar', 'Beauty Salon', 'Bed & Breakfast', 'Best Rates', 'Best Value', 'Bike Rental', 'Boat Rental', 'Body Treatments', 'Boutique Hotel', 'Bowling', 'Breakfast Included', 'Bungalows', 'Business Centre', 'Business Hotel', 'Cable TV', 'Camping Site', 'Car Park', 'Casa Rural (ES)', 'Casino (Hotel)', 'Central Heating', 'Cheap', 'Childcare', 'Club Hotel', 'Computer with Internet', 'Concierge', 'Conference Rooms', 'Convenience Store', 'Convention Hotel', 'Cosmetic Mirror', 'Cot', 'Country Hotel', 'Deals + Beach (AR)', 'Deals + Beach (DE)', 'Deals + Beach (DK)', 'Deals + Beach (GR)', 'Deals + Beach (IT)', 'Deals + Beach (JP)', 'Deals + Beach (MX)', 'Deals + Beach (NL/BE)', 'Deals + Beach (PT)', 'Deals + Beach (TR)', 'Deck Chairs', 'Design Hotel', 'Desk', 'Direct beach access', 'Disneyland', 'Disneyland Paris', 'Diving', 'Doctor On-Site', 'Eco-Friendly hotel', 'Electric Kettle', 'Excellent Rating', 'Express Check-In / Check-Out', 'Family Friendly', 'Fan', 'Farmstay', 'Fitness', 'Flatscreen TV', 'Focus on Distance', 'Focus on Rating', 'Free WiFi (Combined)', 'Free WiFi (Public Areas)', 'Free WiFi (Rooms)', 'Fridge', 'From 2 Stars', 'From 3 Stars', 'From 4 Stars', 'Gay Friendly', 'Golf Course', 'Good Rating', 'Guest House', 'Gym', 'Hairdresser', 'Hairdryer', 'Halal Food', 'Hammam', 'Health Retreat', 'Hiking Trail', 'Holiday', 'Honeymoon', 'Horse Riding', 'Hostal (ES)', 'Hostel', 'Hot Stone Massage', 'Hotel', 'Hotel Bar', 'Hotel Chain', 'House / Apartment', 'Hydrotherapy', 'Hypoallergenic Bedding', 'Hypoallergenic Rooms', 'Internet (Rooms)', 'Ironing Board', 'Jacuzzi (Hotel)', "Kids' Club", 'Kitchen', 'Kosher Food', 'Large Groups', 'Large Hotel', 'Last Minute', 'Laundry Service', 'Lift', 'Luxury Hotel', 'Massage', 'Microwave', 'Mid-Size Hotel', 'Minigolf', 'Motel', 'Next Friday', 'Next Monday', 'Next Saturday', 'Next Sunday', 'Next Weekend', 'Nightclub', 'Non-Smoking Rooms', 'OFF - Rating Good', 'OFF - Rating Very Good', 'On-Site Boutique Shopping', 'Onsen', 'Openable Windows', 'Organised Activities', 'Pay-TV', 'Pet Friendly', 'Playground', 'Pool Table', 'Porter', 'Pousada (BR)', 'Radio', 'Reception (24/7)', 'Resort', 'Restaurant', 'Romantic', 'Room Service', 'Room Service (24/7)', 'Safe (Hotel)', 'Safe (Rooms)', 'Sailing', 'Satellite TV', 'Satisfactory Rating', 'Sauna', 'Self Catering', 'Senior Travellers', 'Serviced Apartment', 'Shooting Sports', 'Shower', 'Singles', 'Sitting Area (Rooms)', 'Ski Resort', 'Skiing', 'Small Hotel', 'Solarium', 'Sort By Distance', 'Sort By Popularity', 'Sort By Rating', 'Sort by Price', 'Spa (Wellness Facility)', 'Spa Hotel', 'Steam Room', 'Sun Umbrellas', 'Surfing', 'Swimming Pool (Bar)', 'Swimming Pool (Combined Filter)', 'Swimming Pool (Indoor)', 'Swimming Pool (Outdoor)', 'Szep Kartya', 'Table Tennis', 'Telephone', 'Teleprinter', 'Television', 'Tennis Court', 'Tennis Court (Indoor)', 'Terrace (Hotel)', 'Theme Hotel', 'This Monday', 'This Weekend', 'Today', 'Tomorrow', 'Top Deals', 'Towels', 'Very Good Rating', 'Volleyball', 'Washing Machine', 'Water Slide', 'Wheelchair Accessible', 'WiFi (Public Areas)', 'WiFi (Rooms)']
device_list = ['desktop', 'mobile', 'tablet']
platform_list = ['AA', 'AE', 'AR', 'AT', 'AU', 'BE', 'BG', 'BR', 'CA', 'CH', 'CL', 'CN', 'CO', 'CZ', 'DE', 'DK', 'EC', 'ES', 'FI', 'FR', 'GR', 'HK', 'HR', 'HU', 'ID', 'IE', 'IL', 'IN', 'IT', 'JP', 'KR', 'MX', 'MY', 'NL', 'NO', 'NZ', 'PE', 'PH', 'PL', 'PT', 'RO', 'RS', 'RU', 'SE', 'SG', 'SI', 'SK', 'TH', 'TR', 'TW', 'UK', 'US', 'UY', 'VN', 'ZA']

submission_header = ['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']

class header_idx(enum.Enum):
    USER_ID = 0
    SESSION_ID = 1
    TIMESTAMP = 2
    STEP = 3
    ACTION_TYPE = 4
    REFERENCE = 5
    PLATFORM = 6
    CITY = 7
    DEVICE = 8
    CURRENT_FILTERS = 9
    IMPRESSIONS = 10
    PRICES = 11

    ITEM_ID = 0
    ITEM_PROPERTIES = 1

class interaction_type(enum.Enum):
    RATING = 0
    INFO = 1
    IMAGE = 2
    DEALS = 3

class feature_size(enum.Enum):
    FILTER = 205 
    INTERACTION = 4 
    DEVICE = 3
    PLATFORM = 55

    ITEM_PROPERTIES = 157
    PRICE = 1
    DISPLAY_ORDER = 1
    ITEM_ID = 1
    USER_ID = 1

    RATING = 1

    SESSION = FILTER+INTERACTION+DEVICE+PLATFORM
    #SESSION = FILTER+INTERACTION
    ITEM = ITEM_PROPERTIES+PRICE
    Y = RATING

    #WHOLE_SIZE = SESSION+ITEM+Y
    WHOLE_SIZE = ITEM_ID+INTERACTION+PRICE+DISPLAY_ORDER+Y

class data_size(enum.Enum):
    TRAIN_EXAMPLE = 15932992
    ITEM_ID = 927142
    USER_ID = 948041
    TRAIN_SESSION = 648663
    VAL_SESSION = 6587

class encoding_idx(enum.Enum) :
    ITEM_ID =0 
    PRICE =1 
    DP_ORDER =2 
    INTERACTION = slice(3, 7)
    Y = 7

    '''
    ITEM_ID = slice(0, 1)
    PRICE = slice(1, 2)
    DP_ORDER = slice(2, 3)
    INTERACTION = slice(3, 7)
    Y = slice(7, 8)
    '''

impression_enc_path ='/home/sap/class/ml/final/data/pkl/train_val_encoding.hdf5' 
impression_enc_train_name = 'train_enc'
impression_enc_val_name = 'val_enc'
impression_dic_train_path = '/data1/sap/ml/final/data/pkl/train_dic.pkl'
impression_dic_val_path = '/data1/sap/ml/final/data/pkl/val_dic.pkl'
test_enc_path ='/home/sap/class/ml/final/data/pkl/test_encoding.hdf5'
enc_test_name = 'train_enc'
test_dic_path = '/data1/sap/ml/final/data/pkl/test_dic.pkl'

item_property_binary_enc_path = '/data1/sap/ml/final/data/pkl/feature_list/item_proprerty_encoding.npy'

platform_list_path = '/data1/sap/ml/final/data/pkl/feature_list/platform_list.pkl'
device_list_path = '/data1/sap/ml/final/data/pkl/feature_list/device_list.pkl'
filter_list_path = '/data1/sap/ml/final/data/pkl/feature_list/filter_list.pkl'
user_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/user_id_list.pkl'
session_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/session_id_list.pkl'
item_id_list_path = '/data1/sap/ml/final/data/pkl/feature_list/item_id_list.pkl'

W_user_id_path = '/data1/sap/ml/final/data/pkl/W_user_id.np'
B_user_id_path = '/data1/sap/ml/final/data/pkl/B_user_id.np'
W_item_id_path = '/data1/sap/ml/final/data/pkl/W_item_id.np'
B_item_id_path = '/data1/sap/ml/final/data/pkl/B_item_id.np'

model_path = '/data1/sap/ml/final/model'
sumission_csv_path = '/home/sap/class/ml/final/submission/submission3.csv' 
