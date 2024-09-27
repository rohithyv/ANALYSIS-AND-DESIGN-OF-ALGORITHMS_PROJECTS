import time
import numpy as np
import math
import matplotlib.pyplot as plt

# Insertion sort for small groups
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# QuickSelect using Median of Medians
def quickselect(arr, k):
    if len(arr) <= 5:
        arr = insertion_sort(arr)
        return arr[k]

    # Step 1: Divide into groups of 5 and sort each group
    groups = [arr[i:i + 5] for i in range(0, len(arr), 5)]
    medians = [insertion_sort(group)[len(group) // 2] for group in groups]

    # Step 2: Find the median of medians
    median_of_medians = quickselect(medians, len(medians) // 2)

    # Step 3: Partition the array
    low = [x for x in arr if x < median_of_medians]
    high = [x for x in arr if x > median_of_medians]

    # Step 4: Recursively call quickselect
    if k < len(low):
        return quickselect(low, k)
    elif k > len(arr) - len(high):
        return quickselect(high, k - (len(arr) - len(high)))
    else:
        return median_of_medians

# Function to measure the time in nanoseconds
def measure_time(arr, k):
    start_time = time.perf_counter_ns()
    result = quickselect(arr, k)
    end_time = time.perf_counter_ns()
    execution_time = end_time - start_time
    return execution_time

# Theoretical time complexity for QuickSelect (linear)
def theoretical_time_fun(n):
    return n  # Since QuickSelect with Median of Medians is O(n)

# Generate random arrays and measure execution times
n_values = [10**i for i in range(2, 7)]  # Array sizes from 10^2 to 10^6
experimental_times = []
theoretical_times = []
scaling_constant = 0

for n in n_values:
    arr = np.random.randint(1, 1000, n)  # Random array of size n
    k = len(arr) // 2  # Median element index

    # Measure experimental time
    exp_time = measure_time(arr, k)
    experimental_times.append(exp_time)

    # Compute theoretical time
    theoretical_time = theoretical_time_fun(n)
    theoretical_times.append(theoretical_time)

# Step 2b: Compute scaling constant
scaling_constant = np.mean(experimental_times) / np.mean(theoretical_times)

# Adjust theoretical times using the scaling constant
adjusted_theoretical_times = [scaling_constant * t for t in theoretical_times]

# Create a table for comparison
table_data = list(zip(n_values, experimental_times, theoretical_times, [scaling_constant] * len(n_values), adjusted_theoretical_times))

# Print the table
print(f"{'n':<10}{'Experimental Result (ns)':<30}{'Theoretical Result':<20}{'Scaling Constant':<20}{'Adjusted Theoretical Result'}")
for row in table_data:
    print(f"{row[0]:<10}{row[1]:<30}{row[2]:<20}{row[3]:<20}{row[4]}")

# Step 3: Plot the two series (experimental vs. adjusted theoretical result)
plt.figure(figsize=(10, 6))
plt.plot(n_values, experimental_times, label='Experimental Times (ns)', marker='o')
plt.plot(n_values, adjusted_theoretical_times, label='Adjusted Theoretical Times', marker='x')

# Scale to log-log for easier comparison
plt.xscale('log')
plt.yscale('log')

plt.title('Comparison of Experimental and Adjusted Theoretical Time Complexities')
plt.xlabel('Array Size n (log scale)')
plt.ylabel('Time (log scale)')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
