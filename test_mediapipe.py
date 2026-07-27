import sys
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}\n")

print("Attempting to import mediapipe...")
try:
    import mediapipe as mp
    print("SUCCESS: mediapipe imported.")
    print(f"Type of imported 'mp' object: {type(mp)}")
    print(f"File location of 'mp' module: {mp.__file__}\n")

    print("Inspecting attributes of the 'mp' module:")
    # Print the first 20 attributes to see what's actually in there
    attributes = dir(mp)
    for i, attr in enumerate(attributes[:20]):
        print(f"  - {attr}")
    
    if len(attributes) > 20:
        print(f"  ... and {len(attributes) - 20} more.")

    print("\n--- Checking for 'solutions' attribute ---")
    if hasattr(mp, 'solutions'):
        print("SUCCESS: 'mp.solutions' attribute found.")
    else:
        print("!!! FAILURE: 'mp.solutions' attribute is missing !!!")

except Exception as e:
    print(f"\n!!! UNEXPECTED FAILURE !!!")
    print(f"An unexpected error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()