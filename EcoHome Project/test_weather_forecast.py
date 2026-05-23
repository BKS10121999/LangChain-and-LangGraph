#!/usr/bin/env python
"""Test script for get_weather_forecast tool"""

from tools import get_weather_forecast
import json

print("=" * 80)
print("TESTING get_weather_forecast TOOL")
print("=" * 80)

# Test 1: Default call
print("\n[Test 1] Default call (today, 3 days)")
print("-" * 80)
result = get_weather_forecast.invoke({})
if "error" not in result:
    print(f"✓ Date range: {result['date_range']}")
    print(f"✓ Days forecast: {result['days_forecast']}")
    print(f"✓ Generated at: {result['generated_at']}")
    print(f"\nFirst day summary:")
    summary = result['forecasts'][0]['day_summary']
    print(f"  Temperature: {summary['temperature_min_c']}°C to {summary['temperature_max_c']}°C")
    print(f"  Cloud cover: {summary['cloud_cover_percent']}%")
    print(f"  Humidity: {summary['humidity_avg_percent']}%")
    print(f"  Weather: {summary['weather_condition']}")
    print(f"  Solar factor: {summary['solar_generation_factor']}")
else:
    print(f"✗ Error: {result['error']}")

# Test 2: Specific date with 5 days
print("\n[Test 2] Specific date (2026-05-25) with 5 days")
print("-" * 80)
result = get_weather_forecast.invoke({'date': '2026-05-25', 'days': 5})
if "error" not in result:
    print(f"✓ Date range: {result['date_range']}")
    print(f"✓ Days forecast: {result['days_forecast']}")
    for i, forecast in enumerate(result['forecasts']):
        summary = forecast['day_summary']
        print(f"  Day {i+1}: {forecast['date']} - {summary['weather_condition']}, solar={summary['solar_generation_factor']}")
else:
    print(f"✗ Error: {result['error']}")

# Test 3: Invalid date format
print("\n[Test 3] Invalid date format (should handle gracefully)")
print("-" * 80)
result = get_weather_forecast.invoke({'date': '25-05-2026'})
if "error" in result:
    print(f"✓ Error handled correctly: {result['error']}")
else:
    print(f"✗ Should have returned error for invalid date")

# Test 4: Hourly data structure
print("\n[Test 4] Hourly data verification")
print("-" * 80)
result = get_weather_forecast.invoke({'date': '2026-05-23'})
if "error" not in result:
    hourly = result['forecasts'][0]['hourly_data']
    print(f"✓ Total hourly records: {len(hourly)}")
    print(f"✓ Hour range: {hourly[0]['hour']} to {hourly[-1]['hour']}")
    
    # Check for noon (solar peak)
    noon = hourly[12]
    print(f"\nNoon (12:00) data:")
    print(f"  Temperature: {noon['temperature_c']}°C")
    print(f"  Cloud cover: {noon['cloud_cover_percent']}%")
    print(f"  Solar factor: {noon['solar_generation_factor']}")
    print(f"  Humidity: {noon['humidity_percent']}%")
    print(f"  Condition: {noon['weather_condition']}")
    
    # Check morning vs afternoon (solar should be higher at noon)
    morning = hourly[9]
    afternoon = hourly[15]
    print(f"\nComparative solar generation:")
    print(f"  9 AM:  {morning['solar_generation_factor']}")
    print(f"  12 PM: {noon['solar_generation_factor']} (peak)")
    print(f"  3 PM:  {afternoon['solar_generation_factor']}")
else:
    print(f"✗ Error: {result['error']}")

# Test 5: Solar generation factor ranges
print("\n[Test 5] Solar generation factor realistic ranges")
print("-" * 80)
result = get_weather_forecast.invoke({'date': '2026-05-24', 'days': 3})
if "error" not in result:
    for forecast in result['forecasts']:
        summary = forecast['day_summary']
        max_solar = max([h['solar_generation_factor'] for h in forecast['hourly_data']])
        min_solar = min([h['solar_generation_factor'] for h in forecast['hourly_data']])
        print(f"✓ {forecast['date']} ({summary['weather_condition']})")
        print(f"    Daily solar factor: {summary['solar_generation_factor']} (range: {min_solar} - {max_solar})")
else:
    print(f"✗ Error: {result['error']}")

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED")
print("=" * 80)
