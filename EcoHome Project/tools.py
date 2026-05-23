"""
Tools for EcoHome Energy Advisor Agent
"""
import os
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy import func
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models.energy import DatabaseManager, EnergyUsage, SolarGeneration

# Initialize database manager
db_manager = DatabaseManager()

@tool
def get_weather_forecast(date: str = None, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast for a specific date and number of subsequent days.
    
    Returns realistic mock weather data suitable for predicting solar generation 
    and HVAC usage patterns. The forecast includes hourly and daily summaries 
    with temperature, cloud cover, solar generation factors, humidity, and 
    weather conditions.
    
    Args:
        date (str): Start date in YYYY-MM-DD format (defaults to today). 
                   Used to generate forecasts from this date forward.
        days (int): Number of days to forecast (1-7, default 3). 
                   Forecasts consecutive days starting from the provided date.
    
    Returns:
        Dict[str, Any]: Comprehensive weather forecast data with structure:
            {
                "date_range": str,
                "generated_at": str (ISO format),
                "days_forecast": int,
                "forecasts": [
                    {
                        "date": str (YYYY-MM-DD),
                        "day_summary": {
                            "temperature_min_c": float,
                            "temperature_max_c": float,
                            "temperature_avg_c": float,
                            "cloud_cover_percent": int (0-100),
                            "humidity_avg_percent": int (0-100),
                            "weather_condition": str ("sunny", "partly_cloudy", "cloudy", "rainy"),
                            "solar_generation_factor": float (0.0-1.0)
                        },
                        "hourly_data": [
                            {
                                "hour": int (0-23),
                                "temperature_c": float,
                                "cloud_cover_percent": int,
                                "solar_generation_factor": float,
                                "humidity_percent": int,
                                "weather_condition": str
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
    
    Raises:
        Returns error dict if date format is invalid or date is in the past.
    
    Example:
        >>> forecast = get_weather_forecast("2026-05-24", days=3)
        >>> print(forecast["forecasts"][0]["day_summary"]["solar_generation_factor"])
        0.75  # Good solar generation day
        
        >>> forecast = get_weather_forecast()  # Uses today's date
    """
    try:
        # Parse and validate date
        if date is None:
            forecast_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            try:
                forecast_start = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": f"Invalid date format: '{date}'. Use YYYY-MM-DD format.",
                    "example": "2026-05-24"
                }
        
        # Validate days parameter
        days = max(1, min(int(days), 7))  # Clamp between 1-7
        
        # Initialize forecast data
        forecasts = []
        
        # Generate realistic weather patterns
        # Simulate seasonal and daily variations
        base_temp = 22 + 5 * random.gauss(0, 1)  # Base temperature with random variation
        
        for day_offset in range(days):
            current_date = forecast_start + timedelta(days=day_offset)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Generate daily weather pattern (some consistency day-to-day)
            daily_weather_seed = random.random()
            if daily_weather_seed < 0.35:
                daily_condition = "sunny"
                cloud_base = 10
                solar_factor_base = 0.85
            elif daily_weather_seed < 0.70:
                daily_condition = "partly_cloudy"
                cloud_base = 40
                solar_factor_base = 0.60
            elif daily_weather_seed < 0.90:
                daily_condition = "cloudy"
                cloud_base = 70
                solar_factor_base = 0.25
            else:
                daily_condition = "rainy"
                cloud_base = 90
                solar_factor_base = 0.05
            
            # Daily temperature variation (cooler/warmer trend)
            daily_temp_trend = -3 + 6 * random.random()
            day_temp_min = base_temp + daily_temp_trend - 5
            day_temp_max = base_temp + daily_temp_trend + 8
            
            # Generate hourly data for the day
            hourly_data = []
            temps = []
            clouds = []
            humidities = []
            solar_factors = []
            
            for hour in range(24):
                # Temperature follows diurnal cycle
                hour_factor = (hour - 6) / 18  # Normalized to roughly 6 AM - midnight
                hour_sine = max(-1, min(1, 2 * (hour_factor - 0.5)))
                temp_at_hour = day_temp_min + (day_temp_max - day_temp_min) * (0.5 + 0.5 * hour_sine)
                temp_at_hour += random.gauss(0, 0.5)  # Add hourly noise
                temps.append(temp_at_hour)
                
                # Cloud cover varies slightly throughout day
                cloud_variation = cloud_base + random.randint(-15, 15)
                cloud_variation = max(0, min(100, cloud_variation))
                clouds.append(cloud_variation)
                
                # Humidity inversely correlates with temperature (typical pattern)
                humidity = 50 - (temp_at_hour - day_temp_min) * 20 / (day_temp_max - day_temp_min) + random.randint(-10, 10)
                humidity = max(20, min(95, humidity))
                humidities.append(int(humidity))
                
                # Solar generation factor based on hour and cloud cover
                # Peak solar between 9 AM and 3 PM
                if 6 <= hour <= 18:
                    hours_from_noon = abs(hour - 12)
                    solar_peak_factor = max(0, 1 - (hours_from_noon / 6.5) ** 2)
                    cloud_reduction = (100 - cloud_variation) / 100
                    solar_factor = solar_factor_base * solar_peak_factor * cloud_reduction * random.uniform(0.85, 1.0)
                else:
                    solar_factor = 0.0
                
                solar_factors.append(round(max(0, solar_factor), 3))
                
                # Determine condition for the hour (similar to daily, with variations)
                if cloud_variation < 25:
                    hour_condition = "sunny"
                elif cloud_variation < 50:
                    hour_condition = "partly_cloudy"
                elif cloud_variation < 80:
                    hour_condition = "cloudy"
                else:
                    hour_condition = "rainy"
                
                hourly_data.append({
                    "hour": hour,
                    "temperature_c": round(temp_at_hour, 1),
                    "cloud_cover_percent": cloud_variation,
                    "solar_generation_factor": solar_factors[-1],
                    "humidity_percent": humidities[-1],
                    "weather_condition": hour_condition
                })
            
            # Calculate daily summary
            day_summary = {
                "temperature_min_c": round(min(temps), 1),
                "temperature_max_c": round(max(temps), 1),
                "temperature_avg_c": round(sum(temps) / len(temps), 1),
                "cloud_cover_percent": round(sum(clouds) / len(clouds)),
                "humidity_avg_percent": round(sum(humidities) / len(humidities)),
                "weather_condition": daily_condition,
                "solar_generation_factor": round(sum(solar_factors) / len(solar_factors), 3)
            }
            
            forecasts.append({
                "date": date_str,
                "day_summary": day_summary,
                "hourly_data": hourly_data
            })
        
        # Build response
        end_date = forecast_start + timedelta(days=days - 1)
        date_range = f"{forecast_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        return {
            "date_range": date_range,
            "generated_at": datetime.now().isoformat(),
            "days_forecast": days,
            "forecasts": forecasts
        }
        
    except Exception as e:
        return {
            "error": f"Error generating weather forecast: {str(e)}",
            "error_type": type(e).__name__
        } 

@tool
def get_electricity_prices(date: str = None) -> Dict[str, Any]:
    """
    Get simulated dynamic electricity prices for a specific date or current day.

    Produces hourly time-of-use pricing with weekday/weekend logic and realistic
    evening spikes so the energy advisor can recommend cheaper appliance windows.
    
    Args:
        date (str): Date in YYYY-MM-DD format (defaults to today)
    
    Returns:
        Dict[str, Any]: Structured JSON-compatible pricing data with hourly
        rates, pricing periods, demand charges, and appliance scheduling hints.
    """
    try:
        if date is None:
            target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": f"Invalid date format: '{date}'. Use YYYY-MM-DD format.",
                    "example": "2026-05-24"
                }

        is_weekend = target_date.weekday() >= 5
        day_type = "weekend" if is_weekend else "weekday"

        # Stable seed keeps the same date consistent while still looking dynamic.
        rng = random.Random(target_date.strftime("%Y-%m-%d"))

        if is_weekend:
            period_config = {
                "off_peak": {"rate": 0.105, "demand_charge": 0.00},
                "mid_peak": {"rate": 0.155, "demand_charge": 0.01},
                "peak": {"rate": 0.245, "demand_charge": 0.035},
            }
        else:
            period_config = {
                "off_peak": {"rate": 0.115, "demand_charge": 0.00},
                "mid_peak": {"rate": 0.185, "demand_charge": 0.02},
                "peak": {"rate": 0.315, "demand_charge": 0.055},
            }

        hourly_rates = []
        cheapest_hours = []
        avoid_hours = []

        for hour in range(24):
            if is_weekend:
                if 0 <= hour <= 6 or 22 <= hour <= 23:
                    period = "off_peak"
                elif 16 <= hour <= 20:
                    period = "peak"
                else:
                    period = "mid_peak"
            else:
                if 0 <= hour <= 5 or 23 <= hour <= 23:
                    period = "off_peak"
                elif 7 <= hour <= 10 or 14 <= hour <= 16 or 21 <= hour <= 22:
                    period = "mid_peak"
                else:
                    period = "peak"

            config = period_config[period]
            rate = config["rate"]
            demand_charge = config["demand_charge"]

            # Evening household demand creates the most expensive spike window.
            spike_multiplier = 1.0
            spike_reason = None
            if 17 <= hour <= 20:
                if is_weekend:
                    spike_multiplier = rng.uniform(1.12, 1.32)
                else:
                    spike_multiplier = rng.uniform(1.22, 1.55)
                spike_reason = "evening_demand_spike"
            elif 6 <= hour <= 8 and not is_weekend:
                spike_multiplier = rng.uniform(1.05, 1.16)
                spike_reason = "morning_ramp"

            dynamic_variation = rng.uniform(-0.008, 0.012)
            final_rate = max(0.075, (rate * spike_multiplier) + dynamic_variation)
            estimated_total_rate = final_rate + demand_charge

            if period == "off_peak":
                cheapest_hours.append(hour)
            if period == "peak" or spike_reason == "evening_demand_spike":
                avoid_hours.append(hour)

            hourly_rates.append({
                "hour": hour,
                "time_window": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
                "period": period,
                "rate_usd_per_kwh": round(final_rate, 3),
                "demand_charge_usd_per_kwh": round(demand_charge, 3),
                "estimated_total_usd_per_kwh": round(estimated_total_rate, 3),
                "price_signal": (
                    "best_for_flexible_loads" if period == "off_peak"
                    else "acceptable_for_moderate_loads" if period == "mid_peak"
                    else "avoid_for_flexible_loads"
                ),
                "spike_reason": spike_reason
            })

        sorted_by_price = sorted(hourly_rates, key=lambda item: item["estimated_total_usd_per_kwh"])
        recommended_windows = [
            {
                "start_hour": sorted_by_price[i]["hour"],
                "time_window": sorted_by_price[i]["time_window"],
                "estimated_total_usd_per_kwh": sorted_by_price[i]["estimated_total_usd_per_kwh"],
                "recommended_for": ["dishwasher", "laundry", "ev_charging", "water_heater"]
            }
            for i in range(min(6, len(sorted_by_price)))
        ]

        average_rate = sum(item["estimated_total_usd_per_kwh"] for item in hourly_rates) / len(hourly_rates)
        peak_rate = max(item["estimated_total_usd_per_kwh"] for item in hourly_rates)
        lowest_rate = min(item["estimated_total_usd_per_kwh"] for item in hourly_rates)

        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "pricing_type": "simulated_dynamic_time_of_use",
            "currency": "USD",
            "unit": "per_kWh",
            "day_type": day_type,
            "is_weekend": is_weekend,
            "summary": {
                "lowest_rate_usd_per_kwh": round(lowest_rate, 3),
                "average_rate_usd_per_kwh": round(average_rate, 3),
                "highest_rate_usd_per_kwh": round(peak_rate, 3),
                "cheapest_hours": cheapest_hours,
                "hours_to_avoid": sorted(set(avoid_hours)),
                "optimization_hint": (
                    "Schedule flexible appliances during off-peak overnight hours and avoid evening spike periods."
                )
            },
            "period_definitions": {
                "off_peak": "Lowest-cost hours; best for flexible appliance scheduling.",
                "mid_peak": "Moderate-cost hours; acceptable when off-peak scheduling is not practical.",
                "peak": "Highest-cost hours; avoid running flexible high-load appliances."
            },
            "recommended_appliance_windows": recommended_windows,
            "hourly_rates": hourly_rates
        }
    except Exception as e:
        return {
            "error": f"Error generating electricity prices: {str(e)}",
            "error_type": type(e).__name__
        }

@tool
def query_energy_usage(start_date: str, end_date: str, device_type: str = None) -> Dict[str, Any]:
    """
    Query energy usage data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        device_type (str): Optional device type filter (e.g., "EV", "HVAC", "appliance")
    
    Returns:
        Dict[str, Any]: Energy usage data with consumption details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_usage_by_date_range(start_dt, end_dt)
        
        if device_type:
            records = [r for r in records if r.device_type == device_type]
        
        usage_data = {
            "start_date": start_date,
            "end_date": end_date,
            "device_type": device_type,
            "total_records": len(records),
            "total_consumption_kwh": round(sum(r.consumption_kwh for r in records), 2),
            "total_cost_usd": round(sum(r.cost_usd or 0 for r in records), 2),
            "records": []
        }
        
        for record in records:
            usage_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": record.consumption_kwh,
                "device_type": record.device_type,
                "device_name": record.device_name,
                "cost_usd": record.cost_usd
            })
        
        return usage_data
    except Exception as e:
        return {"error": f"Failed to query energy usage: {str(e)}"}

@tool
def query_historical_energy_usage(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    device: str = None,
    device_type: str = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Query historical energy usage and costs from the SQLite database.

    Supports filtering by a single date, date range, device name, and device
    type. Returns structured summaries with aggregate statistics, peak usage
    periods, average consumption, cost totals, and simple trend analysis.

    Args:
        date (str): Optional single date in YYYY-MM-DD format.
        start_date (str): Optional start date in YYYY-MM-DD format.
        end_date (str): Optional end date in YYYY-MM-DD format.
        device (str): Optional device name filter. Partial matches are allowed.
        device_type (str): Optional device type filter, such as EV, HVAC, or appliance.
        limit (int): Maximum number of detailed records to return, from 1 to 500.

    Returns:
        Dict[str, Any]: Structured JSON-compatible usage summary.
    """
    session = None
    try:
        if date and (start_date or end_date):
            return {
                "error": "Use either 'date' for a single day or 'start_date'/'end_date' for a range, not both."
            }

        if date:
            try:
                start_dt = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": f"Invalid date format: '{date}'. Use YYYY-MM-DD format.",
                    "example": "2026-05-24"
                }
            end_dt = start_dt + timedelta(days=1)
        else:
            if start_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    return {
                        "error": f"Invalid start_date format: '{start_date}'. Use YYYY-MM-DD format.",
                        "example": "2026-05-01"
                    }
            else:
                start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=30)

            if end_date:
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                except ValueError:
                    return {
                        "error": f"Invalid end_date format: '{end_date}'. Use YYYY-MM-DD format.",
                        "example": "2026-05-31"
                    }
            else:
                end_dt = datetime.now()

        if end_dt <= start_dt:
            return {"error": "end_date must be later than start_date."}

        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError):
            return {"error": "limit must be an integer between 1 and 500."}

        session = db_manager.get_session()
        query = session.query(EnergyUsage).filter(
            EnergyUsage.timestamp >= start_dt,
            EnergyUsage.timestamp < end_dt
        )

        if device:
            query = query.filter(EnergyUsage.device_name.ilike(f"%{device}%"))
        if device_type:
            query = query.filter(EnergyUsage.device_type == device_type)

        records = query.order_by(EnergyUsage.timestamp).all()

        total_records = len(records)
        total_consumption = sum(record.consumption_kwh for record in records)
        total_cost = sum(record.cost_usd or 0 for record in records)
        average_consumption = total_consumption / total_records if total_records else 0
        average_cost = total_cost / total_records if total_records else 0
        effective_rate = total_cost / total_consumption if total_consumption else 0

        device_rows = session.query(
            EnergyUsage.device_name,
            EnergyUsage.device_type,
            func.count(EnergyUsage.id),
            func.sum(EnergyUsage.consumption_kwh),
            func.avg(EnergyUsage.consumption_kwh),
            func.sum(func.coalesce(EnergyUsage.cost_usd, 0))
        ).filter(
            EnergyUsage.timestamp >= start_dt,
            EnergyUsage.timestamp < end_dt
        )

        if device:
            device_rows = device_rows.filter(EnergyUsage.device_name.ilike(f"%{device}%"))
        if device_type:
            device_rows = device_rows.filter(EnergyUsage.device_type == device_type)

        device_rows = device_rows.group_by(
            EnergyUsage.device_name,
            EnergyUsage.device_type
        ).order_by(func.sum(EnergyUsage.consumption_kwh).desc()).all()

        by_device = [
            {
                "device_name": row[0] or "unknown",
                "device_type": row[1] or "unknown",
                "records": row[2],
                "total_consumption_kwh": round(row[3] or 0, 2),
                "average_consumption_kwh": round(row[4] or 0, 2),
                "total_cost_usd": round(row[5] or 0, 2)
            }
            for row in device_rows
        ]

        daily_totals = {}
        hourly_totals = {}
        for record in records:
            day_key = record.timestamp.strftime("%Y-%m-%d")
            hour_key = record.timestamp.hour

            if day_key not in daily_totals:
                daily_totals[day_key] = {"consumption_kwh": 0, "cost_usd": 0, "records": 0}
            daily_totals[day_key]["consumption_kwh"] += record.consumption_kwh
            daily_totals[day_key]["cost_usd"] += record.cost_usd or 0
            daily_totals[day_key]["records"] += 1

            if hour_key not in hourly_totals:
                hourly_totals[hour_key] = {"consumption_kwh": 0, "cost_usd": 0, "records": 0}
            hourly_totals[hour_key]["consumption_kwh"] += record.consumption_kwh
            hourly_totals[hour_key]["cost_usd"] += record.cost_usd or 0
            hourly_totals[hour_key]["records"] += 1

        daily_summary = [
            {
                "date": day,
                "total_consumption_kwh": round(values["consumption_kwh"], 2),
                "total_cost_usd": round(values["cost_usd"], 2),
                "records": values["records"]
            }
            for day, values in sorted(daily_totals.items())
        ]

        peak_hourly_periods = [
            {
                "hour": hour,
                "time_window": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
                "total_consumption_kwh": round(values["consumption_kwh"], 2),
                "average_consumption_kwh": round(values["consumption_kwh"] / values["records"], 2),
                "total_cost_usd": round(values["cost_usd"], 2),
                "records": values["records"]
            }
            for hour, values in sorted(
                hourly_totals.items(),
                key=lambda item: item[1]["consumption_kwh"],
                reverse=True
            )[:5]
        ]

        peak_records = sorted(records, key=lambda record: record.consumption_kwh, reverse=True)[:5]
        peak_usage_events = [
            {
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": round(record.consumption_kwh, 2),
                "cost_usd": round(record.cost_usd or 0, 2),
                "device_name": record.device_name,
                "device_type": record.device_type
            }
            for record in peak_records
        ]

        trend_direction = "flat"
        trend_percent_change = 0
        if len(daily_summary) >= 2:
            midpoint = max(1, len(daily_summary) // 2)
            first_period = daily_summary[:midpoint]
            second_period = daily_summary[midpoint:]
            first_avg = sum(day["total_consumption_kwh"] for day in first_period) / len(first_period)
            second_avg = sum(day["total_consumption_kwh"] for day in second_period) / max(1, len(second_period))
            if first_avg:
                trend_percent_change = ((second_avg - first_avg) / first_avg) * 100
            if trend_percent_change > 5:
                trend_direction = "increasing"
            elif trend_percent_change < -5:
                trend_direction = "decreasing"

        record_details = [
            {
                "timestamp": record.timestamp.isoformat(),
                "consumption_kwh": round(record.consumption_kwh, 2),
                "cost_usd": round(record.cost_usd or 0, 2),
                "device_type": record.device_type,
                "device_name": record.device_name
            }
            for record in records[:limit]
        ]

        return {
            "query": {
                "date": date,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": (end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
                "device": device,
                "device_type": device_type,
                "record_limit": limit
            },
            "summary": {
                "total_records": total_records,
                "total_consumption_kwh": round(total_consumption, 2),
                "average_consumption_kwh": round(average_consumption, 2),
                "total_cost_usd": round(total_cost, 2),
                "average_cost_usd": round(average_cost, 2),
                "effective_rate_usd_per_kwh": round(effective_rate, 3)
            },
            "aggregate_statistics": {
                "by_device": by_device,
                "daily_totals": daily_summary
            },
            "peak_usage": {
                "peak_hourly_periods": peak_hourly_periods,
                "peak_usage_events": peak_usage_events
            },
            "trend_summary": {
                "direction": trend_direction,
                "percent_change": round(trend_percent_change, 1),
                "basis": "Compares average daily consumption in the first half of the query window to the second half."
            },
            "records_returned": len(record_details),
            "records": record_details
        }
    except Exception as e:
        return {
            "error": f"Failed to query historical energy usage: {str(e)}",
            "error_type": type(e).__name__
        }
    finally:
        if session is not None:
            session.close()

@tool
def query_solar_generation(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Query solar generation data from the database for a specific date range.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        Dict[str, Any]: Solar generation data with production details
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        
        records = db_manager.get_generation_by_date_range(start_dt, end_dt)
        
        generation_data = {
            "start_date": start_date,
            "end_date": end_date,
            "total_records": len(records),
            "total_generation_kwh": round(sum(r.generation_kwh for r in records), 2),
            "average_daily_generation": round(sum(r.generation_kwh for r in records) / max(1, (end_dt - start_dt).days), 2),
            "records": []
        }
        
        for record in records:
            generation_data["records"].append({
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": record.generation_kwh,
                "weather_condition": record.weather_condition,
                "battery_storage_level": record.battery_storage_level,
                "exported_to_grid_kwh": record.exported_to_grid_kwh
            })
        
        return generation_data
    except Exception as e:
        return {"error": f"Failed to query solar generation: {str(e)}"}

@tool
def analyze_solar_generation(
    date: str = None,
    start_date: str = None,
    end_date: str = None,
    weather_condition: str = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Analyze historical solar generation, battery storage, and grid exports.

    Queries the SQLite solar_generation table using SQLAlchemy and returns
    structured summaries for production, battery storage trends, sunny vs cloudy
    generation, peak solar periods, and export-to-grid behavior.

    Args:
        date (str): Optional single date in YYYY-MM-DD format.
        start_date (str): Optional start date in YYYY-MM-DD format.
        end_date (str): Optional end date in YYYY-MM-DD format.
        weather_condition (str): Optional weather filter such as sunny, cloudy, or night.
        limit (int): Maximum number of detailed records to return, from 1 to 500.

    Returns:
        Dict[str, Any]: Structured JSON-compatible solar analysis.
    """
    session = None
    try:
        if date and (start_date or end_date):
            return {
                "error": "Use either 'date' for a single day or 'start_date'/'end_date' for a range, not both."
            }

        if date:
            try:
                start_dt = datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                return {
                    "error": f"Invalid date format: '{date}'. Use YYYY-MM-DD format.",
                    "example": "2026-05-24"
                }
            end_dt = start_dt + timedelta(days=1)
        else:
            if start_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    return {
                        "error": f"Invalid start_date format: '{start_date}'. Use YYYY-MM-DD format.",
                        "example": "2026-05-01"
                    }
            else:
                start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=30)

            if end_date:
                try:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
                except ValueError:
                    return {
                        "error": f"Invalid end_date format: '{end_date}'. Use YYYY-MM-DD format.",
                        "example": "2026-05-31"
                    }
            else:
                end_dt = datetime.now()

        if end_dt <= start_dt:
            return {"error": "end_date must be later than start_date."}

        try:
            limit = max(1, min(int(limit), 500))
        except (TypeError, ValueError):
            return {"error": "limit must be an integer between 1 and 500."}

        session = db_manager.get_session()
        query = session.query(SolarGeneration).filter(
            SolarGeneration.timestamp >= start_dt,
            SolarGeneration.timestamp < end_dt
        )

        if weather_condition:
            query = query.filter(SolarGeneration.weather_condition == weather_condition)

        records = query.order_by(SolarGeneration.timestamp).all()
        total_records = len(records)
        total_generation = sum(record.generation_kwh for record in records)
        total_exported = sum(record.exported_to_grid_kwh or 0 for record in records)
        battery_values = [
            record.battery_storage_level
            for record in records
            if record.battery_storage_level is not None
        ]
        generating_records = [record for record in records if record.generation_kwh > 0]

        average_generation = total_generation / total_records if total_records else 0
        average_active_generation = total_generation / len(generating_records) if generating_records else 0
        average_battery = sum(battery_values) / len(battery_values) if battery_values else 0
        min_battery = min(battery_values) if battery_values else 0
        max_battery = max(battery_values) if battery_values else 0
        export_ratio = total_exported / total_generation if total_generation else 0

        hourly_totals = {}
        daily_totals = {}
        weather_totals = {}
        battery_deltas = []

        previous_battery = None
        for record in records:
            day_key = record.timestamp.strftime("%Y-%m-%d")
            hour_key = record.timestamp.hour
            weather_key = record.weather_condition or "unknown"

            if day_key not in daily_totals:
                daily_totals[day_key] = {
                    "generation_kwh": 0,
                    "exported_to_grid_kwh": 0,
                    "battery_total": 0,
                    "battery_records": 0,
                    "records": 0
                }
            daily_totals[day_key]["generation_kwh"] += record.generation_kwh
            daily_totals[day_key]["exported_to_grid_kwh"] += record.exported_to_grid_kwh or 0
            daily_totals[day_key]["records"] += 1
            if record.battery_storage_level is not None:
                daily_totals[day_key]["battery_total"] += record.battery_storage_level
                daily_totals[day_key]["battery_records"] += 1

            if hour_key not in hourly_totals:
                hourly_totals[hour_key] = {
                    "generation_kwh": 0,
                    "exported_to_grid_kwh": 0,
                    "records": 0
                }
            hourly_totals[hour_key]["generation_kwh"] += record.generation_kwh
            hourly_totals[hour_key]["exported_to_grid_kwh"] += record.exported_to_grid_kwh or 0
            hourly_totals[hour_key]["records"] += 1

            if weather_key not in weather_totals:
                weather_totals[weather_key] = {
                    "generation_kwh": 0,
                    "exported_to_grid_kwh": 0,
                    "records": 0
                }
            weather_totals[weather_key]["generation_kwh"] += record.generation_kwh
            weather_totals[weather_key]["exported_to_grid_kwh"] += record.exported_to_grid_kwh or 0
            weather_totals[weather_key]["records"] += 1

            if previous_battery is not None and record.battery_storage_level is not None:
                battery_deltas.append(record.battery_storage_level - previous_battery)
            if record.battery_storage_level is not None:
                previous_battery = record.battery_storage_level

        daily_summary = [
            {
                "date": day,
                "total_generation_kwh": round(values["generation_kwh"], 2),
                "exported_to_grid_kwh": round(values["exported_to_grid_kwh"], 2),
                "average_battery_storage_level": round(
                    values["battery_total"] / values["battery_records"], 2
                ) if values["battery_records"] else 0,
                "records": values["records"]
            }
            for day, values in sorted(daily_totals.items())
        ]

        weather_comparison = [
            {
                "weather_condition": weather,
                "records": values["records"],
                "total_generation_kwh": round(values["generation_kwh"], 2),
                "average_generation_kwh": round(values["generation_kwh"] / values["records"], 2),
                "exported_to_grid_kwh": round(values["exported_to_grid_kwh"], 2)
            }
            for weather, values in sorted(
                weather_totals.items(),
                key=lambda item: item[1]["generation_kwh"],
                reverse=True
            )
        ]

        daylight_generation_values = [
            record.generation_kwh
            for record in records
            if 6 <= record.timestamp.hour <= 18 and record.generation_kwh > 0
        ]
        inferred_sunny_avg = None
        inferred_cloudy_avg = None
        inferred_method = None
        if daylight_generation_values:
            sorted_daylight = sorted(daylight_generation_values)
            sunny_threshold = sorted_daylight[int((len(sorted_daylight) - 1) * 0.75)]
            cloudy_threshold = sorted_daylight[int((len(sorted_daylight) - 1) * 0.35)]
            inferred_sunny_values = [
                record.generation_kwh
                for record in records
                if 6 <= record.timestamp.hour <= 18 and record.generation_kwh >= sunny_threshold
            ]
            inferred_cloudy_values = [
                record.generation_kwh
                for record in records
                if 6 <= record.timestamp.hour <= 18 and 0 < record.generation_kwh <= cloudy_threshold
            ]
            inferred_sunny_avg = (
                sum(inferred_sunny_values) / len(inferred_sunny_values)
                if inferred_sunny_values else None
            )
            inferred_cloudy_avg = (
                sum(inferred_cloudy_values) / len(inferred_cloudy_values)
                if inferred_cloudy_values else None
            )
            inferred_method = "Inferred from daylight generation quartiles because explicit sunny/cloudy labels were unavailable."

        sunny_avg = next(
            (item["average_generation_kwh"] for item in weather_comparison if item["weather_condition"] == "sunny"),
            None
        )
        cloudy_avg = next(
            (item["average_generation_kwh"] for item in weather_comparison if item["weather_condition"] == "cloudy"),
            None
        )
        comparison_sunny_avg = sunny_avg if sunny_avg is not None else inferred_sunny_avg
        comparison_cloudy_avg = cloudy_avg if cloudy_avg is not None else inferred_cloudy_avg
        sunny_vs_cloudy = {
            "sunny_average_generation_kwh": round(comparison_sunny_avg, 2) if comparison_sunny_avg is not None else None,
            "cloudy_average_generation_kwh": round(comparison_cloudy_avg, 2) if comparison_cloudy_avg is not None else None,
            "sunny_generation_lift_percent": round(((comparison_sunny_avg - comparison_cloudy_avg) / comparison_cloudy_avg) * 100, 1)
            if comparison_sunny_avg is not None and comparison_cloudy_avg not in (None, 0) else None,
            "comparison_method": "explicit_weather_labels" if sunny_avg is not None and cloudy_avg is not None else inferred_method
        }

        peak_solar_periods = [
            {
                "hour": hour,
                "time_window": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
                "total_generation_kwh": round(values["generation_kwh"], 2),
                "average_generation_kwh": round(values["generation_kwh"] / values["records"], 2),
                "exported_to_grid_kwh": round(values["exported_to_grid_kwh"], 2),
                "records": values["records"]
            }
            for hour, values in sorted(
                hourly_totals.items(),
                key=lambda item: item[1]["generation_kwh"],
                reverse=True
            )[:5]
        ]

        peak_events = [
            {
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": round(record.generation_kwh, 2),
                "weather_condition": record.weather_condition,
                "battery_storage_level": round(record.battery_storage_level or 0, 2),
                "exported_to_grid_kwh": round(record.exported_to_grid_kwh or 0, 2)
            }
            for record in sorted(records, key=lambda item: item.generation_kwh, reverse=True)[:5]
        ]

        battery_trend = "flat"
        average_battery_change = sum(battery_deltas) / len(battery_deltas) if battery_deltas else 0
        if average_battery_change > 0.05:
            battery_trend = "charging"
        elif average_battery_change < -0.05:
            battery_trend = "discharging"

        if export_ratio > 0.25 and average_battery < 80:
            battery_recommendation = "Increase battery charging priority during peak solar hours before exporting surplus to the grid."
        elif min_battery < 20:
            battery_recommendation = "Reserve more battery capacity overnight or reduce evening discharge to avoid low storage periods."
        elif export_ratio > 0.4:
            battery_recommendation = "Consider shifting flexible loads into peak solar hours or expanding storage to capture frequent surplus."
        else:
            battery_recommendation = "Battery behavior looks balanced; continue using high-load appliances during peak solar production windows."

        export_by_hour = [
            {
                "hour": hour,
                "time_window": f"{hour:02d}:00-{(hour + 1) % 24:02d}:00",
                "exported_to_grid_kwh": round(values["exported_to_grid_kwh"], 2)
            }
            for hour, values in sorted(
                hourly_totals.items(),
                key=lambda item: item[1]["exported_to_grid_kwh"],
                reverse=True
            )[:5]
        ]

        record_details = [
            {
                "timestamp": record.timestamp.isoformat(),
                "generation_kwh": round(record.generation_kwh, 2),
                "weather_condition": record.weather_condition,
                "battery_storage_level": round(record.battery_storage_level or 0, 2),
                "exported_to_grid_kwh": round(record.exported_to_grid_kwh or 0, 2)
            }
            for record in records[:limit]
        ]

        return {
            "query": {
                "date": date,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": (end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
                "weather_condition": weather_condition,
                "record_limit": limit
            },
            "summary_statistics": {
                "total_records": total_records,
                "total_generation_kwh": round(total_generation, 2),
                "average_generation_kwh": round(average_generation, 2),
                "average_active_generation_kwh": round(average_active_generation, 2),
                "total_exported_to_grid_kwh": round(total_exported, 2),
                "export_ratio": round(export_ratio, 3),
                "average_battery_storage_level": round(average_battery, 2),
                "min_battery_storage_level": round(min_battery, 2),
                "max_battery_storage_level": round(max_battery, 2)
            },
            "peak_solar_periods": {
                "top_hours": peak_solar_periods,
                "top_events": peak_events
            },
            "weather_comparison": {
                "by_condition": weather_comparison,
                "sunny_vs_cloudy": sunny_vs_cloudy
            },
            "battery_analysis": {
                "trend": battery_trend,
                "average_hourly_change": round(average_battery_change, 2),
                "recommendation": battery_recommendation
            },
            "export_to_grid_analysis": {
                "total_exported_kwh": round(total_exported, 2),
                "export_ratio": round(export_ratio, 3),
                "highest_export_hours": export_by_hour,
                "recommendation": (
                    "Move flexible appliance loads into high-export solar hours to self-consume more generation."
                    if total_exported > 0 else
                    "No grid exports detected in this period; prioritize preserving battery charge for evening demand."
                )
            },
            "daily_totals": daily_summary,
            "records_returned": len(record_details),
            "records": record_details
        }
    except Exception as e:
        return {
            "error": f"Failed to analyze solar generation: {str(e)}",
            "error_type": type(e).__name__
        }
    finally:
        if session is not None:
            session.close()

@tool
def get_recent_energy_summary(hours: int = 24) -> Dict[str, Any]:
    """
    Get a summary of recent energy usage and solar generation.
    
    Args:
        hours (int): Number of hours to look back (default 24)
    
    Returns:
        Dict[str, Any]: Summary of recent energy data
    """
    try:
        usage_records = db_manager.get_recent_usage(hours)
        generation_records = db_manager.get_recent_generation(hours)
        
        summary = {
            "time_period_hours": hours,
            "usage": {
                "total_consumption_kwh": round(sum(r.consumption_kwh for r in usage_records), 2),
                "total_cost_usd": round(sum(r.cost_usd or 0 for r in usage_records), 2),
                "device_breakdown": {}
            },
            "generation": {
                "total_generation_kwh": round(sum(r.generation_kwh for r in generation_records), 2),
                "average_weather": "sunny" if generation_records else "unknown"
            }
        }
        
        # Calculate device breakdown
        for record in usage_records:
            device = record.device_type or "unknown"
            if device not in summary["usage"]["device_breakdown"]:
                summary["usage"]["device_breakdown"][device] = {
                    "consumption_kwh": 0,
                    "cost_usd": 0,
                    "records": 0
                }
            summary["usage"]["device_breakdown"][device]["consumption_kwh"] += record.consumption_kwh
            summary["usage"]["device_breakdown"][device]["cost_usd"] += record.cost_usd or 0
            summary["usage"]["device_breakdown"][device]["records"] += 1
        
        # Round the breakdown values
        for device_data in summary["usage"]["device_breakdown"].values():
            device_data["consumption_kwh"] = round(device_data["consumption_kwh"], 2)
            device_data["cost_usd"] = round(device_data["cost_usd"], 2)
        
        return summary
    except Exception as e:
        return {"error": f"Failed to get recent energy summary: {str(e)}"}

@tool
def search_energy_tips(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for energy-saving tips and best practices using RAG.
    
    Args:
        query (str): Search query for energy tips
        max_results (int): Maximum number of results to return
    
    Returns:
        Dict[str, Any]: Relevant energy tips and best practices
    """
    try:
        # Initialize vector store if it doesn't exist
        persist_directory = "data/vectorstore"
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        # Load documents if vector store doesn't exist
        if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
            # Load documents
            documents = []
            for doc_path in ["data/documents/tip_device_best_practices.txt", "data/documents/tip_energy_savings.txt"]:
                if os.path.exists(doc_path):
                    loader = TextLoader(doc_path)
                    docs = loader.load()
                    documents.extend(docs)
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory
            )
        else:
            # Load existing vector store
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=max_results)
        
        results = {
            "query": query,
            "total_results": len(docs),
            "tips": []
        }
        
        for i, doc in enumerate(docs):
            results["tips"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "relevance_score": "high" if i < 2 else "medium" if i < 4 else "low"
            })
        
        return results
    except Exception as e:
        return {"error": f"Failed to search energy tips: {str(e)}"}

@tool
def calculate_energy_savings(device_type: str, current_usage_kwh: float, 
                           optimized_usage_kwh: float, price_per_kwh: float = 0.12) -> Dict[str, Any]:
    """
    Calculate potential energy savings from optimization.
    
    Args:
        device_type (str): Type of device being optimized
        current_usage_kwh (float): Current energy usage in kWh
        optimized_usage_kwh (float): Optimized energy usage in kWh
        price_per_kwh (float): Price per kWh (default 0.12)
    
    Returns:
        Dict[str, Any]: Savings calculation results
    """
    savings_kwh = current_usage_kwh - optimized_usage_kwh
    savings_usd = savings_kwh * price_per_kwh
    savings_percentage = (savings_kwh / current_usage_kwh) * 100 if current_usage_kwh > 0 else 0
    
    return {
        "device_type": device_type,
        "current_usage_kwh": current_usage_kwh,
        "optimized_usage_kwh": optimized_usage_kwh,
        "savings_kwh": round(savings_kwh, 2),
        "savings_usd": round(savings_usd, 2),
        "savings_percentage": round(savings_percentage, 1),
        "price_per_kwh": price_per_kwh,
        "annual_savings_usd": round(savings_usd * 365, 2)
    }


TOOL_KIT = [
    get_weather_forecast,
    get_electricity_prices,
    query_energy_usage,
    query_historical_energy_usage,
    query_solar_generation,
    analyze_solar_generation,
    get_recent_energy_summary,
    search_energy_tips,
    calculate_energy_savings
]
