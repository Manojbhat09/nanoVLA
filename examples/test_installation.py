#!/usr/bin/env python3
"""
Test script to verify nanoVLA installation works correctly
"""

def test_basic_import():
    """Test basic nanovla import"""
    try:
        import sys
        sys.path.insert(0, 'nanovla/src')
        import nanovla
        print("✅ nanovla imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import nanovla: {e}")
        return False

def test_nanovla_model():
    """Test nanoVLA model availability"""
    try:
        import sys
        sys.path.insert(0, 'nanovla/src')
        import nanovla
        if hasattr(nanovla, 'nanoVLA'):
            print("✅ nanoVLA model class available")
            return True
        else:
            print("❌ nanoVLA model class not found")
            return False
    except Exception as e:
        print(f"❌ Error testing nanoVLA model: {e}")
        return False

def test_from_vlm():
    """Test from_vlm functionality"""
    try:
        import sys
        sys.path.insert(0, 'nanovla/src')
        import nanovla
        if hasattr(nanovla, 'nanoVLA') and hasattr(nanovla.nanoVLA, 'from_vlm'):
            print("✅ from_vlm method available")
            return True
        else:
            print("❌ from_vlm method not found")
            return False
    except Exception as e:
        print(f"❌ Error testing from_vlm: {e}")
        return False

def test_vlm_registry():
    """Test VLM registry"""
    try:
        from nanovla.vlm_factory import vlm_registry
        available_vlms = vlm_registry.list_available()
        print(f"✅ VLM registry working. Available VLMs: {available_vlms}")
        return True
    except Exception as e:
        print(f"❌ Error testing VLM registry: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing nanoVLA Installation")
    print("=" * 40)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("nanoVLA Model", test_nanovla_model), 
        ("from_vlm Method", test_from_vlm),
        ("VLM Registry", test_vlm_registry),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📋 Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("🎉 All tests passed! nanoVLA is ready to use.")
    else:
        print("⚠️  Some tests failed. Installation may need fixing.")

if __name__ == "__main__":
    main()
