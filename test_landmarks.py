#!/usr/bin/env python3
"""Test landmark parsing to verify format understanding"""

def test_landmark_parsing():
    # Test with a sample line from the file
    test_line = "img/img_00000001.jpg 1 1 1 92 19 1 148 21 0 51 64 1 169 78 0 69 201 0 160 204 0 0 0 0 0 0"
    parts = test_line.strip().split()
    
    print(f"Total parts: {len(parts)}")
    print(f"Image name: {parts[0]}")
    print(f"Clothes type: {parts[1]}")
    print(f"Variation type: {parts[2]}")
    
    print("\nLandmarks:")
    for i in range(8):
        base_idx = 3 + i * 3
        if base_idx + 2 < len(parts):
            visibility = int(parts[base_idx])
            x = int(parts[base_idx + 1])
            y = int(parts[base_idx + 2])
            print(f"Landmark {i+1}: visibility={visibility}, x={x}, y={y}")
        else:
            print(f"Landmark {i+1}: Not enough data")
    
    # Let's also check what landmarks are visible
    print("\nVisible landmarks:")
    visible_count = 0
    for i in range(8):
        base_idx = 3 + i * 3
        if base_idx < len(parts):
            visibility = int(parts[base_idx])
            if visibility == 1 and base_idx + 2 < len(parts):
                x = int(parts[base_idx + 1])
                y = int(parts[base_idx + 2])
                if x > 0 and y > 0:
                    visible_count += 1
                    print(f"Landmark {i+1}: x={x}, y={y}")
    
    print(f"\nTotal visible landmarks: {visible_count}")

if __name__ == "__main__":
    test_landmark_parsing()