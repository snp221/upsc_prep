#!/usr/bin/env python3
"""
Test script to verify ChromaDB search functionality is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_build import UPSCRAGSystem

def test_search_functionality():
    """Test the search functionality for each collection"""
    
    print("Initializing UPSC RAG System...")
    try:
        rag = UPSCRAGSystem()
        print("✓ RAG system initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize RAG system: {e}")
        return False
    
    # Test queries for different collections
    test_queries = {
        'essays': 'democracy and governance',
        'books': 'Indian constitution',
        'ncert': 'ancient history',
        'pyq': 'administrative reforms'
    }
    
    print("\n" + "="*50)
    print("Testing Search Functionality")
    print("="*50)
    
    all_tests_passed = True
    
    for collection, query in test_queries.items():
        print(f"\nTesting {collection} collection with query: '{query}'")
        try:
            # Search specific collection
            results = rag.search(query, n_results=3, source_filter=[collection])
            
            if results['total_found'] > 0:
                print(f"✓ Found {results['total_found']} results")
                for i, result in enumerate(results['results'][:2], 1):
                    doc_preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
                    print(f"  {i}. Distance: {result['distance']:.4f}")
                    print(f"     Preview: {doc_preview}")
            else:
                print(f"⚠ No results found for '{query}' in {collection}")
                
        except Exception as e:
            print(f"✗ Error searching {collection}: {e}")
            all_tests_passed = False
    
    # Test general search across all collections
    print(f"\n{'='*50}")
    print("Testing General Search (All Collections)")
    print("="*50)
    
    general_query = "Indian political system"
    print(f"\nTesting general search with query: '{general_query}'")
    
    try:
        results = rag.search(general_query, n_results=5)
        print(f"✓ Found {results['total_found']} results across all collections")
        
        # Show breakdown by collection
        collection_counts = {}
        for result in results['results']:
            collection = result['collection']
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
        
        for collection, count in collection_counts.items():
            print(f"  - {collection}: {count} results")
            
    except Exception as e:
        print(f"✗ Error in general search: {e}")
        all_tests_passed = False
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("✓ All tests passed! Search functionality is working properly.")
    else:
        print("✗ Some tests failed. Check the errors above.")
    print("="*50)
    
    return all_tests_passed

if __name__ == "__main__":
    test_search_functionality()