import pandas as pd
import h3
import ast

def clean_and_index_data(file_path: str, h3_resolution: int = 8) -> pd.DataFrame:
    """
    Takes raw traffic data as input and then removes noise, filters multicollinearity,
    and applies Uber H3 spatial indexing with severity scoring.
    """
    df = pd.read_csv(file_path)
    df.drop(columns = ['description','closed_datetime','action_taken_timestamp', 'created_by_id', 'device_id', 'data_sent_to_scita_timestamp'], inplace = True)
    
    if 'validation_status' in df.columns:
        df = df[df['validation_status'] == 'approved']
        
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'])   
 
    if 'offence_code' in df.columns:
        df = df.drop(columns=['offence_code'])
        
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # 2. Domain-Specific Severity Scoring (Section 4)
    severity_weights = {
        # CRITICAL IMPACT: Creates an immediate physical chokepoint
        'DOUBLE PARKING': 5.0,
        'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 5.0,
        
        # HIGH IMPACT: Blocks arterial flow or critical safety/transit zones
        'PARKING IN A MAIN ROAD': 4.0,
        'PARKING NEAR ROAD CROSSING': 4.0,
        'PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS': 4.0,
        'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 4.0,
        
        # MODERATE IMPACT: Standard illegal parking, causes general friction
        'WRONG PARKING': 2.0,
        'NO PARKING': 2.0,
        'PARKING OTHER THAN BUS STOP': 2.0,
        
        # LOW IMPACT: Blocks pedestrians, but vehicles can still generally pass
        'PARKING ON FOOTPATH': 1.0,
        
        # ZERO CONGESTION IMPACT: Administrative or behavioral offenses
        'DEFECTIVE NUMBER PLATE': 0.0,
        'REFUSE TO GO FOR HIRE': 0.0,
        'USING BLACK FILM/OTHER MATERIALS': 0.0,
        'DEMANDING EXCESS FARE': 0.0,
        'WITHOUT SIDE MIRROR': 0.0,
        'H T V PROHIBITED': 0.0,
        'OBSTRUCTING DRIVER': 0.0,
        'FAIL TO USE SAFETY BELTS': 0.0,
        'RIDER NOT WEARING HELMET': 0.0,
        '2W/3W - USING MOBILE PHONE': 0.0,
        'OTHER - USING MOBILE PHONE': 0.0,
        
        # MINOR MOVING VIOLATIONS (Caught by static cameras)
        'AGAINST ONE WAY/NO ENTRY': 3.0,
        'VIOLATING LANE DISIPLINE': 3.0,
        'CARRYING LENGHTY MATERIAL': 3.0,
        'JUMPING TRAFFIC SIGNAL': 3.0,
        'U TURN PROHIBITED': 3.0,
        'STOPING ON WHITE/STOP LINE': 3.0
    }
    df['violation_list'] = df['violation_type'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).startswith('[') else [x]
    )
    exploded_df = df.explode('violation_list')
    exploded_df['temp_weight'] = exploded_df['violation_list'].map(severity_weights).fillna(1.0)
    df['severity_score'] = exploded_df.groupby(level=0)['temp_weight'].sum()
    df = df.drop(columns=['violation_list'])
    
    df['h3_cell'] = df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], h3_resolution), 
        axis=1
    )    
    return df

if __name__ == "__main__":
    # Example execution line
    processed_df = clean_and_index_data(r"C:\Users\Admin\Desktop\Round2\data\jan to may police violation_anonymized791b166.csv")
    processed_df.to_csv("data/processed_indexed.csv", index=False)
    print("Date cleaned and processed.")