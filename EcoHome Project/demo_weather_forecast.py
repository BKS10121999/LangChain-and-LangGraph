#!/usr/bin/env python
"""Demonstration of get_weather_forecast tool for energy optimization"""

from tools import get_weather_forecast
import json

print("=" * 80)
print("WEATHER FORECAST FOR SMART ENERGY OPTIMIZATION")
print("=" * 80)

# Scenario: Agent needs to predict solar generation for HVAC scheduling
print("\n📊 SCENARIO: Predict Solar Generation for HVAC Scheduling")
print("-" * 80)

forecast = get_weather_forecast.invoke({'date': '2026-05-24', 'days': 3})

print(f"Forecast period: {forecast['date_range']}")
print(f"Generated: {forecast['generated_at']}\n")

for day_forecast in forecast['forecasts']:
    date = day_forecast['date']
    summary = day_forecast['day_summary']
    
    print(f"📅 {date}")
    print(f"   Weather: {summary['weather_condition'].replace('_', ' ').title()}")
    print(f"   Temperature: {summary['temperature_min_c']}°C to {summary['temperature_max_c']}°C (avg: {summary['temperature_avg_c']}°C)")
    print(f"   Cloud Cover: {summary['cloud_cover_percent']}%")
    print(f"   Humidity: {summary['humidity_avg_percent']}%")
    print(f"   ☀️  Solar Generation Factor: {summary['solar_generation_factor']:.3f}")
    
    # Energy optimization insights
    if summary['solar_generation_factor'] > 0.6:
        recommendation = "EXCELLENT ✓ - Schedule HVAC loads during peak solar hours"
    elif summary['solar_generation_factor'] > 0.3:
        recommendation = "GOOD - Consider moderate HVAC usage during afternoon"
    else:
        recommendation = "POOR - Rely on grid power, minimize HVAC during peak hours"
    
    print(f"   Recommendation: {recommendation}\n")

# Scenario 2: Detailed hourly analysis for demand response
print("\n" + "=" * 80)
print("📈 HOURLY ANALYSIS FOR DEMAND RESPONSE STRATEGY")
print("=" * 80)

forecast = get_weather_forecast.invoke({'date': '2026-05-24', 'days': 1})
hourly = forecast['forecasts'][0]['hourly_data']

print(f"\nDate: {forecast['forecasts'][0]['date']}")
print(f"Weather: {forecast['forecasts'][0]['day_summary']['weather_condition']}\n")

# Find peak solar generation hour
peak_hour = max(hourly, key=lambda x: x['solar_generation_factor'])
print(f"Peak Solar Hour: {peak_hour['hour']:02d}:00 (Factor: {peak_hour['solar_generation_factor']:.3f})")
print(f"Temperature at peak: {peak_hour['temperature_c']}°C")
print(f"Cloud cover: {peak_hour['cloud_cover_percent']}%\n")

# Show hourly comparison table
print("Hourly Solar Generation Forecast:")
print("Hour | Temp(°C) | Cloud(%) | Solar Factor | Humidity(%) | Condition")
print("-" * 70)
for hour in hourly:
    if hour['hour'] % 3 == 0 or hour['hour'] == peak_hour['hour']:  # Show every 3 hours + peak
        print(f"{hour['hour']:2d}:00 |  {hour['temperature_c']:5.1f}  |   {hour['cloud_cover_percent']:2d}%    |    {hour['solar_generation_factor']:.3f}      |     {hour['humidity_percent']:2d}%     | {hour['weather_condition']}")

# Scenario 3: Multi-day planning for energy storage
print("\n" + "=" * 80)
print("🔋 ENERGY STORAGE PLANNING (Next 3 Days)")
print("=" * 80)

forecast = get_weather_forecast.invoke({'date': '2026-05-24', 'days': 3})

total_solar = sum([f['day_summary']['solar_generation_factor'] for f in forecast['forecasts']])
avg_solar = total_solar / len(forecast['forecasts'])

for day_forecast in forecast['forecasts']:
    date = day_forecast['date']
    summary = day_forecast['day_summary']
    solar_factor = summary['solar_generation_factor']
    
    # Estimate battery charging priority
    if solar_factor > avg_solar * 1.2:
        priority = "HIGH ↑↑ - Charge batteries"
    elif solar_factor > avg_solar * 0.8:
        priority = "MEDIUM → - Maintain charge"
    else:
        priority = "LOW ↓↓ - Discharge mode"
    
    status = "✓" if solar_factor > 0.3 else "⚠" if solar_factor > 0.1 else "✗"
    print(f"{status} {date}: Solar={solar_factor:.3f} | Temp: {summary['temperature_avg_c']:5.1f}°C | {priority}")

print(f"\nAverage 3-day solar factor: {avg_solar:.3f}")
print(f"Recommended strategy: {'Solar-dependent' if avg_solar > 0.3 else 'Grid-reliant'} energy management")

print("\n" + "=" * 80)
