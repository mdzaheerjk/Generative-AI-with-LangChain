"""Test client for Ray Serve search deployment."""

import requests
import json


def test_search(query="How can Ray help with deploying LLMs?"):
    """Query the Ray Serve deployment and print results."""
    # URL encode the query
    encoded_query = requests.utils.quote(query)

    # Make the request
    url = f"http://localhost:8000/?query={encoded_query}"
    print(f"Querying: {url}")

    try:
        response = requests.get(url)

        # Check if request was successful
        response.raise_for_status()

        # Parse the JSON response
        results = response.json()["results"]

        # Print the results
        print(f"\nFound {len(results)} results for query: '{query}'\n")

        for i, result in enumerate(results):
            print(f"Result {i + 1}:")
            print(f"Score: {result.get('score', 'N/A')}")
            print(f"Source: {result.get('source', 'Unknown')}")
            print(f"Content: {result.get('content', '')[:200]}...")
            print()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {str(e)}")
    except json.JSONDecodeError:
        print(f"Error parsing response as JSON. Response text: {response.text[:500]}")


if __name__ == "__main__":
    # Test with default query
    test_search()
    test_search("How to use Ray Tune for hyperparameter optimization?")
