"""Test flood tool."""

import numpy as np
import pandas as pd

from pytest import mark

import flood_tool.tool as tool
import test_tool

test_tool = tool.Tool()


def test_lookup_easting_northing():
    """Check"""

    data = test_tool.lookup_easting_northing(['RH16 2QE'])

    assert len(data.index) == 1
    assert 'RH16 2QE' in data.index

    assert np.isclose(data.loc['RH16 2QE', 'easting'], 535295).all()
    assert np.isclose(data.loc['RH16 2QE', 'northing'], 123643).all()


@mark.xfail  # We expect this test to fail until we write some code for it.
def test_lookup_lat_long():
    """Check"""

    data = test_tool.lookup_lat_long(["M34 7QL"])

    assert len(data.index) == 1
    assert 'RH16 2QE' in data.index

    assert np.isclose(data.loc['RH16 2QE', 'latitude'],
                      rtol=1.0e-3).all()
    assert np.isclose(data.loc['RH16 2QE', 'longitude'],
                      rtol=1.0e-3).all()


def test_predict_median_house_price_single():
    """Check single postcode prediction"""

    data = test_tool.predict_median_house_price(['RH16 2QE'])

    assert len(data.index) == 1
    assert 'RH16 2QE' in data.index

    assert np.isclose(data.loc['RH16 2QE', 'median_house_price']).all()

    # Check that the median house price is a float value
    assert isinstance(data.loc['RH16 2QE', 'median_house_price'], float)

    # Check that the median house price is not empty
    assert data.loc['RH16 2QE', 'median_house_price'] != ''

def test_predict_median_house_price_multiple():
    """Check multiple postcodes prediction"""

    postcodes = ['RH16 2QE', 'SW1A 1AA']
    data = test_tool.predict_median_house_price(postcodes)

    assert len(data.index) == len(postcodes)
    for postcode in postcodes:
        assert postcode in data.index
        assert isinstance(data.loc[postcode, 'median_house_price'], float)
        assert data.loc[postcode, 'median_house_price'] != ''

def test_predict_median_house_price_invalid():
    """Check invalid postcode prediction"""

    data = test_tool.predict_median_house_price(['INVALID'])

    assert len(data.index) == 0

def test_predict_median_house_price_empty():
    """Check empty input"""

    data = test_tool.predict_median_house_price([])

    assert len(data.index) == 0

def test_predict_median_house_price_datatypes():
    """Check data types of the returned dataframe"""

    data = test_tool.predict_median_house_price(['RH16 2QE'])

    assert isinstance(data, pd.DataFrame)
    assert data['median_house_price'].dtype == float

def test_predict_median_house_price_performance():
    """Check performance of the function"""

    import time
    start_time = time.time()
    data = test_tool.predict_median_house_price(['RH16 2QE'])
    end_time = time.time()

    assert (end_time - start_time) < 1  # Ensure the function runs within 1 second

def test_predict_median_house_price_consistency():
    """Check consistency of the function"""

    data1 = test_tool.predict_median_house_price(['RH16 2QE'])
    data2 = test_tool.predict_median_house_price(['RH16 2QE'])

    assert data1.equals(data2)
def test_estimate_total_value():
    """Check total value estimation for a sequence of postcodes."""

    # Mock sector data
    test_tool._sector_data = pd.DataFrame({
        'postcodeSector': ['BA1 3', 'BA1 4', 'BA1 5'],
        'numberOfPostcodeUnits': [100, 200, 150],
        'households': [5000, 10000, 7500]
    })

    # Mock split_postcode function
    def mock_split_postcode(postcode):
        return postcode.split(' ')[0] + ' ' + postcode.split(' ')[1][0]

    test_tool.split_postcode = mock_split_postcode

    # Mock predict_flood_class_from_postcode
    test_tool.predict_flood_class_from_postcode = MagicMock(
        return_value=[1, 2, 3]  # Mock flood risk labels for the test postcodes
    )

    # Mock predict_median_house_price
    test_tool.predict_median_house_price = MagicMock(
        side_effect=lambda postcode: 300000 if postcode == 'BA1 3PD' else 400000 if postcode == 'BA1 4XY' else 500000
    )

    # Test postcodes
    postcodes = ['BA1 3PD', 'BA1 4XY', 'BA1 5ZZ']

    # Expected flood probabilities
    flood_probs = [0.001, 0.002, 0.005]

    # Expected results
    expected_rows = []
    for i, sector in enumerate(['BA1 3', 'BA1 4', 'BA1 5']):
        count = test_tool._sector_data[test_tool._sector_data['postcodeSector'] == sector]['numberOfPostcodeUnits'].iloc[0]
        households = test_tool._sector_data[test_tool._sector_data['postcodeSector'] == sector]['households'].iloc[0]
        median_price = test_tool.predict_median_house_price(postcodes[i])
        num_properties = households / count
        total_value = 0.05 * median_price * num_properties * flood_probs[i]
        expected_rows.append({'postcode': postcodes[i], 'total_estimated_value': total_value})

    expected_df = pd.DataFrame(expected_rows).set_index('postcode')

    # Run the method
    result = test_tool.estimate_annual_flood_economic_risk(postcodes)

    # Assert results
    assert len(result) == len(postcodes), "Result does not have the expected number of rows."

    for index, row in result.iterrows():
        expected_value = expected_df.loc[row['postcode'], 'total_estimated_value']
        assert np.isclose(
            row['total_estimated_value'], expected_value, rtol=1e-3
        ), f"Value mismatch for {row['postcode']}."




def test_estimate_annual_human_flood_risk():
    """Check human flood risk estimation for postcodes."""

    # Mock sector data
    test_tool._sector_data = pd.DataFrame({
        'postcodeSector': ['AL1 1', 'AL1 2', 'AL1 3'],
        'numberOfPostcodeUnits': [10, 20, 15],
        'headcount': [1000, 2000, 1500]
    })

    # Mock split_postcode function
    def mock_split_postcode(postcode):
        return postcode.split(' ')[0] + ' ' + postcode.split(' ')[1][0]

    test_tool.split_postcode = mock_split_postcode

    # Mock predict_flood_class_from_postcode
    test_tool.predict_flood_class_from_postcode = MagicMock(
        return_value=[1, 2, 3]  # Mock flood risk labels for the test postcodes
    )

    # Test postcodes
    postcodes = ['AL1 1AA', 'AL1 2BB', 'AL1 3CC']

    # Expected flood probabilities
    flood_probs = [0.001, 0.002, 0.005]

    # Expected results
    expected_rows = []
    for i, sector in enumerate(['AL1 1', 'AL1 2', 'AL1 3']):
        count = test_tool._sector_data[test_tool._sector_data['postcodeSector'] == sector]['numberOfPostcodeUnits'].iloc[0]
        population = test_tool._sector_data[test_tool._sector_data['postcodeSector'] == sector]['headcount'].iloc[0] / count
        total_human_value = 0.1 * population * flood_probs[i]
        expected_rows.append({'postcode': postcodes[i], 'total_esitimated_human_value': total_human_value})

    expected_df = pd.DataFrame(expected_rows).set_index('postcode')

    # Run the method
    result = test_tool.estimate_annual_human_flood_risk(postcodes)

    # Assert results
    assert len(result) == len(postcodes), "Result does not have the expected number of rows."

    assert np.isclose(data.loc[0, 'total_esitimated_human_value'], 0.1 * 1000 * 0.001, rtol=1.0e-3)
    assert np.isclose(data.loc[1, 'total_esitimated_human_value'], 0.1 * 2000 * 0.005, rtol=1.0e-3)

if __name__ == "__main__":
    test_lookup_easting_northing()
    test_lookup_lat_long()
    test_predict_median_house_price_single()
    test_predict_median_house_price_multiple()
    test_predict_median_house_price_invalid()
    test_predict_median_house_price_empty()
    test_predict_median_house_price_datatypes()
    test_predict_median_house_price_performance()
    test_predict_median_house_price_consistency()
