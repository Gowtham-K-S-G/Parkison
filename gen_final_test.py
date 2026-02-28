from model.enhanced_generator import generate_enhanced_parkinson_spiral, generate_enhanced_healthy_spiral

# Generate test images with specific characteristics

# Parkinson's - severe case
generate_enhanced_parkinson_spiral('static/sample_images/test_parkinson_severe.png', seed=999, severity=1.0)
print("Created: test_parkinson_severe.png")

# Parkinson's - mild case
generate_enhanced_parkinson_spiral('static/sample_images/test_parkinson_mild.png', seed=888, severity=0.6)
print("Created: test_parkinson_mild.png")

# Healthy - clear spiral
generate_enhanced_healthy_spiral('static/sample_images/test_healthy_clear.png', seed=777)
print("Created: test_healthy_clear.png")

print("\n✅ Test images generated!")
