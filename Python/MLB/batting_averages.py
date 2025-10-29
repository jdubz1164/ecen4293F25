# Jack Weinheimer
# ECEN 4293 - MLB Batting Average Analysis
# Find the 25 best batting averages per year since 1939 with minimum AB = 500

import csv

def read_csv_to_dict(file_path):
    """
    Reads a CSV file and returns a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: A list of dictionaries containing the CSV data.
    """
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def create_player_name_map(people_data):
    """
    Creates a mapping from playerID to player name.

    Args:
        people_data (list): List of dictionaries from People.csv.

    Returns:
        dict: Dictionary mapping playerID to full name.
    """
    player_map = {}
    for person in people_data:
        player_id = person['playerID']
        # Construct full name from nameFirst and nameLast
        first_name = person.get('nameFirst', '')
        last_name = person.get('nameLast', '')
        full_name = f"{first_name} {last_name}".strip()
        player_map[player_id] = full_name
    return player_map

def calculate_batting_average(hits, at_bats):
    """
    Calculates batting average.

    Args:
        hits (int): Number of hits.
        at_bats (int): Number of at bats.

    Returns:
        float: Batting average (hits / at_bats).
    """
    if at_bats == 0:
        return 0.0
    return hits / at_bats

def find_top_batting_averages(batting_data, player_map, min_ab=500, start_year=1939, top_n=25):
    """
    Finds the top N batting averages per year since a given year with minimum AB requirement.

    Args:
        batting_data (list): List of dictionaries from Batting.csv.
        player_map (dict): Dictionary mapping playerID to full name.
        min_ab (int): Minimum number of at bats required.
        start_year (int): Starting year to consider.
        top_n (int): Number of top averages to return per year.

    Returns:
        dict: Dictionary with year as key and list of top batting average records.
    """
    # Group data by year and filter by minimum AB and start year
    yearly_data = {}

    for row in batting_data:
        year = int(row['yearID'])

        # Filter by start year
        if year < start_year:
            continue

        at_bats = int(row.get('AB', 0))

        # Filter by minimum at bats
        if at_bats < min_ab:
            continue

        hits = int(row.get('H', 0))
        avg = calculate_batting_average(hits, at_bats)

        player_id = row['playerID']
        player_name = player_map.get(player_id, 'Unknown Player')

        record = {
            'playerID': player_id,
            'name': player_name,
            'year': year,
            'AB': at_bats,
            'H': hits,
            'AVG': avg
        }

        if year not in yearly_data:
            yearly_data[year] = []
        yearly_data[year].append(record)

    # Sort each year's data by batting average (descending) and take top N
    top_averages = {}
    for year, records in yearly_data.items():
        sorted_records = sorted(records, key=lambda x: x['AVG'], reverse=True)
        top_averages[year] = sorted_records[:top_n]

    return top_averages

def print_top_batting_averages(top_averages):
    """
    Prints the top batting averages in a formatted way.

    Args:
        top_averages (dict): Dictionary with year as key and list of top records.
    """
    for year in sorted(top_averages.keys()):
        print(f"\n{'='*80}")
        print(f"Year {year} - Top 25 Batting Averages (Minimum 500 AB)")
        print(f"{'='*80}")
        print(f"{'Rank':<6}{'Player Name':<25}{'Player ID':<15}{'AB':<8}{'H':<8}{'AVG':<10}")
        print(f"{'-'*80}")

        for rank, record in enumerate(top_averages[year], start=1):
            print(f"{rank:<6}{record['name']:<25}{record['playerID']:<15}"
                  f"{record['AB']:<8}{record['H']:<8}{record['AVG']:.8f}")

def check_specific_player(top_averages, year, player_id, player_name):
    """
    Checks if a specific player appears in the top averages for a given year.

    Args:
        top_averages (dict): Dictionary with year as key and list of top records.
        year (int): Year to check.
        player_id (str): Player ID to look for.
        player_name (str): Player name for display.

    Returns:
        bool: True if player found in top 25, False otherwise.
    """
    if year not in top_averages:
        print(f"\nNo data found for year {year}")
        return False

    for rank, record in enumerate(top_averages[year], start=1):
        if record['playerID'] == player_id:
            print(f"\n{player_name} found in top 25 for {year}:")
            print(f"Rank: {rank}")
            print(f"Batting Average: {record['AVG']:.8f}")
            print(f"At Bats: {record['AB']}")
            print(f"Hits: {record['H']}")
            return True

    print(f"\n{player_name} (playerID: {player_id}) NOT found in top 25 for {year}")
    print("Possible reasons:")
    print("- Did not have minimum 500 at bats")
    print("- Batting average was not in top 25")
    print("- No record for that year")
    return False

def write_results_to_csv(top_averages, output_path="batting_averages_output.csv"):
    """
    Writes the top batting averages to a CSV file.

    Args:
        top_averages (dict): Dictionary with year as key and list of top records.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, mode='w', newline='') as file:
        fieldnames = ['Year', 'Rank', 'Player_Name', 'Player_ID', 'AB', 'H', 'AVG']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for year in sorted(top_averages.keys()):
            for rank, record in enumerate(top_averages[year], start=1):
                writer.writerow({
                    'Year': year,
                    'Rank': rank,
                    'Player_Name': record['name'],
                    'Player_ID': record['playerID'],
                    'AB': record['AB'],
                    'H': record['H'],
                    'AVG': f"{record['AVG']:.8f}"
                })

    print(f"\nResults written to {output_path}")

def main():
    """Main function to execute the batting average analysis."""
    # File paths (using only csv module, so hardcoded paths)
    batting_path = "Batting.csv"
    people_path = "People.csv"

    print("Reading People.csv...")
    people_data = read_csv_to_dict(people_path)

    print("Creating player name mapping...")
    player_map = create_player_name_map(people_data)

    print("Reading Batting.csv...")
    batting_data = read_csv_to_dict(batting_path)

    print("Calculating top 25 batting averages per year since 1939...")
    top_averages = find_top_batting_averages(batting_data, player_map, min_ab=500, start_year=1939, top_n=25)

    print(f"\nAnalysis complete! Found data for {len(top_averages)} years.")

    # Print all results
    print_top_batting_averages(top_averages)

    # Write results to CSV file
    write_results_to_csv(top_averages)

    # Check for Manny Ramirez in 2008
    print("\n" + "="*80)
    print("Checking for Manny Ramirez in 2008...")
    print("="*80)
    check_specific_player(top_averages, 2008, 'ramirma02', 'Manny Ramirez')

if __name__ == "__main__":
    main()
