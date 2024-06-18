def stable_matching(wanteds: list, preferences: dict):
    result = {}

    for preference_name, preference_numbers in preferences.items():
        for preference_number in preference_numbers:
            if preference_number in wanteds:
                wanteds.remove(preference_number)
                result[preference_name] = (preference_number, preference_numbers.index(preference_number))
                break
        
            current_preference_name_preference_number_index = preference_numbers.index(preference_number)
            for old_result_name, old_result_made in result.items():
                if old_result_made[0] == preference_number and old_result_made[1] > current_preference_name_preference_number_index:
                    result[preference_name] = (preference_number, preference_numbers.index(preference_number))
                    result.pop(old_result_name)
                    break

    for preference_name, preference_numbers in preferences.items():
        if preference_name not in result:
            result[preference_name] = -1
        else:
            result[preference_name] = result[preference_name][0]

    return dict(sorted(result.items()))

wanteds = [35, 76, 88, 76, 27, 98, 29, 88, 88, 71, 88, 93, 26, 26, 8]
preferences = {'A': [34, 76, 59], 'B': [22, 27], 'C': [27, 22], 'D': [23], 'E': [24], 'F': [25], 'G': [26], 'H': [27], 'I': [28], 'J': [29], 'K': [30], 'L': [31], 'M': [32], 'N': [33], 'O': [34]}
result = stable_matching(wanteds, preferences)
print(result)