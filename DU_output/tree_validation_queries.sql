-- Decision Tree Validation Queries
-- Generated from: DU_output/du_model_lg_k_18/3col_sketches_lg_k_18_model_lg_k_18.json
-- Total rules: 32
-- Table: DU_raw
-- Target column: tripOutcome


-- Rule 1: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is FALSE AND booking_type=one_way_trip is FALSE AND city=Pune is FALSE
SELECT
    COUNT(*) as actual_samples,
    8368.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0852 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 0 AND booking_type=one_way_trip = 0 AND city=Pune = 0;


--------------------------------------------------------------------------------


-- Rule 2: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is FALSE AND booking_type=one_way_trip is FALSE AND city=Pune is TRUE
SELECT
    COUNT(*) as actual_samples,
    134.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2015 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 0 AND booking_type=one_way_trip = 0 AND city=Pune = 1;


--------------------------------------------------------------------------------


-- Rule 3: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is FALSE AND booking_type=one_way_trip is TRUE AND dayType=NORMAL_WEEKDAY is FALSE
SELECT
    COUNT(*) as actual_samples,
    2519.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1687 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 0 AND booking_type=one_way_trip = 1 AND dayType=NORMAL_WEEKDAY = 0;


--------------------------------------------------------------------------------


-- Rule 4: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is FALSE AND booking_type=one_way_trip is TRUE AND dayType=NORMAL_WEEKDAY is TRUE
SELECT
    COUNT(*) as actual_samples,
    3391.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1094 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 0 AND booking_type=one_way_trip = 1 AND dayType=NORMAL_WEEKDAY = 1;


--------------------------------------------------------------------------------


-- Rule 5: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is TRUE AND city=Chennai is FALSE AND estimated_usage_bins=LTE_1_HOUR is FALSE
SELECT
    COUNT(*) as actual_samples,
    286.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 1 AND city=Chennai = 0 AND estimated_usage_bins=LTE_1_HOUR = 0;


--------------------------------------------------------------------------------


-- Rule 6: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is TRUE AND city=Chennai is FALSE AND estimated_usage_bins=LTE_1_HOUR is TRUE
SELECT
    COUNT(*) as actual_samples,
    414.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0097 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 1 AND city=Chennai = 0 AND estimated_usage_bins=LTE_1_HOUR = 1;


--------------------------------------------------------------------------------


-- Rule 7: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is FALSE AND sla=Immediate is TRUE AND city=Chennai is TRUE
SELECT
    COUNT(*) as actual_samples,
    65.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0462 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 0 AND sla=Immediate = 1 AND city=Chennai = 1;


--------------------------------------------------------------------------------


-- Rule 8: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is TRUE AND booking_type=one_way_trip is FALSE AND estimated_usage_bins=GT_3_LTE_4_HOURS is FALSE
SELECT
    COUNT(*) as actual_samples,
    97.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1546 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 1 AND booking_type=one_way_trip = 0 AND estimated_usage_bins=GT_3_LTE_4_HOURS = 0;


--------------------------------------------------------------------------------


-- Rule 9: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is TRUE AND booking_type=one_way_trip is FALSE AND estimated_usage_bins=GT_3_LTE_4_HOURS is TRUE
SELECT
    COUNT(*) as actual_samples,
    62.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2419 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 1 AND booking_type=one_way_trip = 0 AND estimated_usage_bins=GT_3_LTE_4_HOURS = 1;


--------------------------------------------------------------------------------


-- Rule 10: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND zone=221 is TRUE AND booking_type=one_way_trip is TRUE
SELECT
    COUNT(*) as actual_samples,
    110.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.4 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND zone=221 = 1 AND booking_type=one_way_trip = 1;


--------------------------------------------------------------------------------


-- Rule 11: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is FALSE AND sla=Immediate is FALSE AND source=android is FALSE AND booking_type=one_way_trip is FALSE
SELECT
    COUNT(*) as actual_samples,
    694.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1138 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 0 AND sla=Immediate = 0 AND source=android = 0 AND booking_type=one_way_trip = 0;


--------------------------------------------------------------------------------


-- Rule 12: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is FALSE AND sla=Immediate is FALSE AND source=android is FALSE AND booking_type=one_way_trip is TRUE
SELECT
    COUNT(*) as actual_samples,
    311.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1833 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 0 AND sla=Immediate = 0 AND source=android = 0 AND booking_type=one_way_trip = 1;


--------------------------------------------------------------------------------


-- Rule 13: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is FALSE AND sla=Immediate is FALSE AND source=android is TRUE AND booking_type=one_way_trip is FALSE
SELECT
    COUNT(*) as actual_samples,
    797.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1844 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 0 AND sla=Immediate = 0 AND source=android = 1 AND booking_type=one_way_trip = 0;


--------------------------------------------------------------------------------


-- Rule 14: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is FALSE AND sla=Immediate is FALSE AND source=android is TRUE AND booking_type=one_way_trip is TRUE
SELECT
    COUNT(*) as actual_samples,
    311.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2669 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 0 AND sla=Immediate = 0 AND source=android = 1 AND booking_type=one_way_trip = 1;


--------------------------------------------------------------------------------


-- Rule 15: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is FALSE AND sla=Immediate is TRUE
SELECT
    COUNT(*) as actual_samples,
    97.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 0 AND sla=Immediate = 1;


--------------------------------------------------------------------------------


-- Rule 16: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is TRUE AND booking_type=outstation is FALSE AND source=android is FALSE AND dayType=NORMAL_WEEKDAY is FALSE
SELECT
    COUNT(*) as actual_samples,
    50.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.36 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 1 AND booking_type=outstation = 0 AND source=android = 0 AND dayType=NORMAL_WEEKDAY = 0;


--------------------------------------------------------------------------------


-- Rule 17: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is TRUE AND booking_type=outstation is FALSE AND source=android is FALSE AND dayType=NORMAL_WEEKDAY is TRUE
SELECT
    COUNT(*) as actual_samples,
    58.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2414 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 1 AND booking_type=outstation = 0 AND source=android = 0 AND dayType=NORMAL_WEEKDAY = 1;


--------------------------------------------------------------------------------


-- Rule 18: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is TRUE AND booking_type=outstation is FALSE AND source=android is TRUE AND estimated_usage_bins=GT_1_LTE_2_HOURS is FALSE
SELECT
    COUNT(*) as actual_samples,
    127.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.3622 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 1 AND booking_type=outstation = 0 AND source=android = 1 AND estimated_usage_bins=GT_1_LTE_2_HOURS = 0;


--------------------------------------------------------------------------------


-- Rule 19: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is TRUE AND booking_type=outstation is FALSE AND source=android is TRUE AND estimated_usage_bins=GT_1_LTE_2_HOURS is TRUE
SELECT
    COUNT(*) as actual_samples,
    59.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.322 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 1 AND booking_type=outstation = 0 AND source=android = 1 AND estimated_usage_bins=GT_1_LTE_2_HOURS = 1;


--------------------------------------------------------------------------------


-- Rule 20: pickUpHourOfDay=VERYLATE is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE AND city=Pune is TRUE AND booking_type=outstation is TRUE
SELECT
    COUNT(*) as actual_samples,
    63.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0952 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1 AND city=Pune = 1 AND booking_type=outstation = 1;


--------------------------------------------------------------------------------


-- Rule 21: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND sla=Immediate is FALSE AND estimated_usage_bins=LTE_1_HOUR is FALSE
SELECT
    COUNT(*) as actual_samples,
    121.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1653 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND sla=Immediate = 0 AND estimated_usage_bins=LTE_1_HOUR = 0;


--------------------------------------------------------------------------------


-- Rule 22: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND sla=Immediate is FALSE AND estimated_usage_bins=LTE_1_HOUR is TRUE
SELECT
    COUNT(*) as actual_samples,
    611.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.1031 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND sla=Immediate = 0 AND estimated_usage_bins=LTE_1_HOUR = 1;


--------------------------------------------------------------------------------


-- Rule 23: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is FALSE AND sla=Immediate is TRUE
SELECT
    COUNT(*) as actual_samples,
    79.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 0 AND sla=Immediate = 1;


--------------------------------------------------------------------------------


-- Rule 24: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is FALSE AND zone_demand_popularity=POPULARITY_INDEX_5 is TRUE
SELECT
    COUNT(*) as actual_samples,
    57.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.3158 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 0 AND zone_demand_popularity=POPULARITY_INDEX_5 = 1;


--------------------------------------------------------------------------------


-- Rule 25: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is TRUE AND zone=216 is FALSE AND zone_demand_popularity=POPULARITY_INDEX_4 is FALSE AND source=ios is FALSE
SELECT
    COUNT(*) as actual_samples,
    488.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2357 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 1 AND zone=216 = 0 AND zone_demand_popularity=POPULARITY_INDEX_4 = 0 AND source=ios = 0;


--------------------------------------------------------------------------------


-- Rule 26: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is TRUE AND zone=216 is FALSE AND zone_demand_popularity=POPULARITY_INDEX_4 is FALSE AND source=ios is TRUE
SELECT
    COUNT(*) as actual_samples,
    523.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2696 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 1 AND zone=216 = 0 AND zone_demand_popularity=POPULARITY_INDEX_4 = 0 AND source=ios = 1;


--------------------------------------------------------------------------------


-- Rule 27: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is TRUE AND zone=216 is FALSE AND zone_demand_popularity=POPULARITY_INDEX_4 is TRUE AND source=ios is FALSE
SELECT
    COUNT(*) as actual_samples,
    91.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.2857 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 1 AND zone=216 = 0 AND zone_demand_popularity=POPULARITY_INDEX_4 = 1 AND source=ios = 0;


--------------------------------------------------------------------------------


-- Rule 28: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is TRUE AND zone=216 is FALSE AND zone_demand_popularity=POPULARITY_INDEX_4 is TRUE AND source=ios is TRUE
SELECT
    COUNT(*) as actual_samples,
    109.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.3853 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 1 AND zone=216 = 0 AND zone_demand_popularity=POPULARITY_INDEX_4 = 1 AND source=ios = 1;


--------------------------------------------------------------------------------


-- Rule 29: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is FALSE AND dayType=WEEKEND is TRUE AND zone=216 is TRUE
SELECT
    COUNT(*) as actual_samples,
    83.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.0723 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 0 AND dayType=WEEKEND = 1 AND zone=216 = 1;


--------------------------------------------------------------------------------


-- Rule 30: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is TRUE AND zone_demand_popularity=POPULARITY_INDEX_2 is FALSE AND source=android is FALSE
SELECT
    COUNT(*) as actual_samples,
    60.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.5833 as expected_positive_rate,
    'Positive' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 1 AND zone_demand_popularity=POPULARITY_INDEX_2 = 0 AND source=android = 0;


--------------------------------------------------------------------------------


-- Rule 31: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is TRUE AND zone_demand_popularity=POPULARITY_INDEX_2 is FALSE AND source=android is TRUE
SELECT
    COUNT(*) as actual_samples,
    51.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.5686 as expected_positive_rate,
    'Positive' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 1 AND zone_demand_popularity=POPULARITY_INDEX_2 = 0 AND source=android = 1;


--------------------------------------------------------------------------------


-- Rule 32: pickUpHourOfDay=VERYLATE is TRUE AND city=Delhi NCR is TRUE AND zone_demand_popularity=POPULARITY_INDEX_2 is TRUE
SELECT
    COUNT(*) as actual_samples,
    57.0 as expected_samples,
    ROUND(AVG(CASE WHEN tripOutcome = 1 THEN 1.0 ELSE 0.0 END), 4) as actual_positive_rate,
    0.4737 as expected_positive_rate,
    'Negative' as expected_prediction
FROM DU_raw
WHERE pickUpHourOfDay=VERYLATE = 1 AND city=Delhi NCR = 1 AND zone_demand_popularity=POPULARITY_INDEX_2 = 1;

