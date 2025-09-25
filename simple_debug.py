#!/usr/bin/env python3
"""Simple debug to understand the batch parsing issue"""

# Simulate what happens in batch_perturb_options_with_phrasedp
test_options = {
    'A': 'Streptococcus viridans',
    'B': 'Enterococcus faecalis',
    'C': 'Staphylococcus epidermidis',
    'D': 'Bacillus cereus'
}

# Step 1: Create combined text (like the function does)
combined_text = ""
for key, value in test_options.items():
    combined_text += f"Option {key}: {value}\n"

print("=== COMBINED TEXT ===")
print(repr(combined_text))
print("\n=== COMBINED TEXT (display) ===")
print(combined_text)

# Step 2: Simulate what PhraseDP might return (based on user's feedback)
# User said: "What type of bacteria are characterized by mauve-colored colonies on an antibiotic-containing medium?"
# This suggests PhraseDP is treating the entire combined text as one sentence and perturbing it as one unit

simulated_perturbed = "What type of bacteria are characterized by mauve-colored colonies on an antibiotic-containing medium?"

print("=== SIMULATED PERTURBED OUTPUT ===")
print(repr(simulated_perturbed))
print(simulated_perturbed)

# Step 3: Try to parse this back (like the function does)
perturbed_options = {}
lines = simulated_perturbed.split('\n')
print(f"\n=== PARSING ATTEMPT ===")
print(f"Lines after split: {lines}")
print(f"Number of lines: {len(lines)}")

for line in lines:
    line = line.strip()
    print(f"  Processing line: '{line}'")
    if line.startswith('Option ') and ':' in line:
        parts = line.split(':', 1)
        if len(parts) == 2:
            option_key = parts[0].replace('Option ', '').strip()
            option_value = parts[1].strip()
            print(f"    Extracted: key='{option_key}', value='{option_value}'")
            if option_key in ['A', 'B', 'C', 'D']:
                perturbed_options[option_key] = option_value
                print(f"    Added to perturbed options")
    else:
        print(f"    No 'Option X:' pattern found")

print(f"\n=== PARSING RESULTS ===")
print(f"Successfully parsed: {len(perturbed_options)} out of {len(test_options)} options")
print(f"Perturbed options: {perturbed_options}")

# Step 4: Show what fallback does
if len(perturbed_options) != len(test_options):
    print(f"\n=== FALLBACK APPROACH ===")
    option_keys = list(test_options.keys())
    lines = [l.strip() for l in simulated_perturbed.split('\n') if l.strip()]
    print(f"Fallback lines: {lines}")

    fallback_result = {}
    for i, key in enumerate(option_keys):
        print(f"  Fallback processing for key: {key}")
        if i < len(lines):
            line = lines[i]
            print(f"    Fallback line: '{line}'")
            if line.startswith(f'Option {key}:'):
                line = line[len(f'Option {key}:'):].strip()
            fallback_result[key] = line
            print(f"    Fallback: set {key} to '{line}'")
        else:
            fallback_result[key] = f"Perturbed option {key}"
            print(f"    Fallback: no line for {key}, using default")

    print(f"Fallback result: {fallback_result}")