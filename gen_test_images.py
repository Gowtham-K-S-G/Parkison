from model.enhanced_generator import generate_enhanced_parkinson_spiral, generate_enhanced_healthy_spiral

# Generate test images
generate_enhanced_parkinson_spiral('static/sample_images/test_pd.png', seed=42, severity=0.9)
generate_enhanced_healthy_spiral('static/sample_images/test_healthy.png', seed=100)
print("Test images created!")
