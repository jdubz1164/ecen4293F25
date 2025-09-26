import csv
from scipy import stats
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def read_csv_data(file_path):
    """
    Reads a CSV file and extracts data into a list of dictionaries.
    
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

def load_arrays_from_data(data):
    """
    Converts list of dictionaries into numpy arrays for analysis.
    
    Args:
        data (list): List of dictionaries containing the CSV data.

    Returns:
        tuple: Two numpy arrays, one for x values and one for y values.
    """
    year = np.array([float(item['yearID']) for item in data], dtype=np.float64)
    team = np.array([str(item['teamID']) for item in data], dtype=np.str_)
    league = np.array([str(item['lgID']) for item in data], dtype=np.str_)
    player = np.array([str(item['playerID']) for item in data], dtype=np.str_)
    salary = np.array([float(item['salary']) for item in data], dtype=np.float64)
    return year, team, league, player, salary

def plot_salaries(file_path):
    """Generate a scatter/line chart for salary data."""
    data = read_csv_data(file_path)
    year, _, _, _, salary = load_arrays_from_data(data)

    # Sort values to get a coherent line plot instead of zig-zagging back and forth in time.
    order = np.argsort(year)
    year_sorted = year[order]
    salary_sorted = salary[order]

    plt.figure(figsize=(10, 6))
    plt.scatter(year, salary, alpha=0.5, label='Salaries')
    plt.plot(year_sorted, salary_sorted, color='orange', linewidth=1, alpha=0.3, label='Salaries (sorted)')
    plt.xlabel('Year')
    plt.ylabel('Salary')
    plt.title('MLB Salaries Over Years by Sean Lahman')
    plt.legend()
    plt.grid()
    plt.savefig('salaries_plot.png')
    plt.show()


if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "Salaries.csv")
    plot_salaries(csv_path)



