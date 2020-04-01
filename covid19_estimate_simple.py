import numpy as np

# start from March 8

confirmed_cases = np.array([0.583, 0.959, 1.3, 1.7, 2.2, 2.7, 3.5, 4.6, 6.4, 7.8, 13.7, 19.1, 25.5, 33.3, 43.8, 53.7, 65.8, 83.8])*1000
estimated_increase_ratio = confirmed_cases[1:]/confirmed_cases[:-1]

estimated_cases = np.zeros(100)
estimated_cases_lower = np.zeros(100)
for i in np.arange(100):
    if i==0:
        estimated_cases[i] = 83800
        estimated_cases_lower[i] = 83800
    else:
        estimated_cases[i] = estimated_cases[i-1]*np.mean(estimated_increase_ratio)
        estimated_cases_lower[i] = estimated_cases_lower[i-1]*1.2


print("estimated ratio:", estimated_increase_ratio)
print("estimated cases using avg ratio", estimated_cases[:18])

for i in range(100):
    if i==30:
        print("="*20)
        print("April 26:")
    if i==60:
        print("May 26:")
    if i==90:
        print("June 26:")
    if i==99:
        print("July 6:")
    print('\tavg est:', estimated_cases[i], "\tlower est:", estimated_cases_lower[i])