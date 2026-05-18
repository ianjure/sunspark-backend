import modal

app = modal.App("sunspark-ocr-backend")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr", "libtesseract-dev")
    .pip_install(
        "fastapi[standard]",
        "python-multipart",
        "pillow",
        "pytesseract",
        "google-genai",
        "requests",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("sunspark-gemini-secret")],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile, Form
    from fastapi.middleware.cors import CORSMiddleware
    from PIL import Image
    from google import genai
    from google.genai import types
    import pytesseract
    import io
    import json
    import os
    import re
    import requests

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # okay for testing; restrict this later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def clean_number(value):
        """
        Converts messy numeric values like '₱ 2,345.67' or '345 kWh'
        into float values.
        """
        if value is None:
            return None

        if isinstance(value, int) or isinstance(value, float):
            return float(value)

        value = str(value)
        cleaned = re.sub(r"[^0-9.]", "", value)

        if cleaned == "":
            return None

        try:
            return float(cleaned)
        except ValueError:
            return None

    def compute_effective_rate(monthly_bill, kwh_usage):
        """
        Computes effective electricity rate based on total bill and kWh usage.

        This is usually better for solar savings estimation than the
        supply/generation rate alone.
        """
        if monthly_bill is not None and kwh_usage is not None and kwh_usage > 0:
            return round(monthly_bill / kwh_usage, 4)

        return None

    def extract_bill_fields_with_llm(ocr_text: str):
        prompt = f"""
You are extracting structured data from OCR text of an electricity bill.

Extract only these fields:
1. monthly_bill - the total amount due or current monthly bill amount
2. kwh_usage - total electricity consumption in kWh
3. rate_per_kwh - the electricity rate per kWh only if clearly shown in the bill

Rules:
- Return null if a field is missing or uncertain.
- Do not guess.
- monthly_bill should be a number only, no currency symbol.
- kwh_usage should be a number only.
- rate_per_kwh should be a number only.
- If multiple bill amounts exist, prefer total amount due, amount due, current amount due, or total current charges.
- If multiple kWh values exist, prefer total kWh used, energy consumption, consumption, or present month usage.
- For rate_per_kwh, only extract a rate directly shown in the bill, such as generation charge rate, supply rate, or energy charge rate.
- Do not compute rate_per_kwh yourself.

OCR TEXT:
{ocr_text}
"""

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "monthly_bill": {
                            "type": "number",
                            "nullable": True,
                        },
                        "kwh_usage": {
                            "type": "number",
                            "nullable": True,
                        },
                        "rate_per_kwh": {
                            "type": "number",
                            "nullable": True,
                        },
                    },
                    "required": [
                        "monthly_bill",
                        "kwh_usage",
                        "rate_per_kwh",
                    ],
                },
            ),
        )

        extracted = json.loads(response.text)

        monthly_bill = clean_number(extracted.get("monthly_bill"))
        kwh_usage = clean_number(extracted.get("kwh_usage"))
        rate_per_kwh_found_on_bill = clean_number(extracted.get("rate_per_kwh"))

        effective_rate_per_kwh = compute_effective_rate(monthly_bill, kwh_usage)

        return {
            "monthly_bill": monthly_bill,
            "kwh_usage": kwh_usage,
            "rate_per_kwh_found_on_bill": rate_per_kwh_found_on_bill,
            "effective_rate_per_kwh": effective_rate_per_kwh,
        }

    def get_nested_value(data, path):
        """
        Safely gets nested values from dictionaries.
        Example:
        get_nested_value(data, ["annual", "data", "GHI"])
        """
        current = data

        for key in path:
            if not isinstance(current, dict):
                return None

            current = current.get(key)

            if current is None:
                return None

        return current

    def get_solar_value(annual_data, key):
        """
        Global Solar Atlas values can sometimes be numbers directly,
        or objects depending on endpoint response shape.
        This normalizes both cases.
        """
        value = annual_data.get(key)

        if value is None:
            return None

        if isinstance(value, int) or isinstance(value, float):
            return float(value)

        if isinstance(value, dict):
            for possible_key in ["value", "annual", "avg", "data"]:
                if possible_key in value:
                    return clean_number(value.get(possible_key))

        return clean_number(value)

    def get_location_name_from_nominatim(lat: float, lon: float):
        """
        Reverse geocodes latitude and longitude using OpenStreetMap Nominatim.

        We use zoom=13 because it usually returns village/suburb-level data,
        which is useful for barangay-level location in the Philippines.
        """
        url = "https://nominatim.openstreetmap.org/reverse"

        response = requests.get(
            url,
            params={
                "format": "jsonv2",
                "lat": lat,
                "lon": lon,
                "zoom": 13,
                "addressdetails": 1,
            },
            timeout=20,
            headers={
                # Use an identifying user agent, not the default python-requests one.
                # Replace the email with your preferred project/contact email.
                "User-Agent": "Sunspark/0.1 (ianjure.data@gmail.com)",
                "Accept": "application/json",
            },
        )

        response.raise_for_status()

        data = response.json()
        address = data.get("address", {})

        # Barangay-level value can appear under different OSM keys.
        barangay = (
            address.get("village")
            or address.get("suburb")
            or address.get("neighbourhood")
            or address.get("quarter")
            or address.get("hamlet")
        )

        # City/municipality can also vary by area.
        city_or_municipality = (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("city_district")
            or address.get("county")
        )

        province = (
            address.get("state")
            or address.get("province")
            or address.get("region")
        )

        return {
            "barangay": barangay,
            "city_or_municipality": city_or_municipality,
            "province": province,
        }
    
    def get_global_solar_atlas_data(lat: float, lon: float):
        """
        Gets long-term average solar resource data from Global Solar Atlas
        for a specific latitude and longitude.

        If official PVOUT is missing, estimates daily PVOUT from annual GHI
        using a simple performance ratio fallback.
        """
        url = "https://api.globalsolaratlas.info/data/lta"

        response = requests.get(
            url,
            params={"loc": f"{lat},{lon}"},
            timeout=20,
            headers={
                "User-Agent": "Sunspark/0.1",
            },
        )

        response.raise_for_status()

        data = response.json()

        annual_data = get_nested_value(data, ["annual", "data"]) or {}

        pvout = get_solar_value(annual_data, "PVOUT")
        ghi = get_solar_value(annual_data, "GHI")
        dni = get_solar_value(annual_data, "DNI")
        dif = get_solar_value(annual_data, "DIF")
        gti = get_solar_value(annual_data, "GTI")
        opta = get_solar_value(annual_data, "OPTA")
        temp = get_solar_value(annual_data, "TEMP")

        atlas_url = f"https://globalsolaratlas.info/map?s={lat},{lon}&m=site"

        ghi_daily = None
        if ghi is not None:
            ghi_daily = round(ghi / 365, 2)

        performance_ratio = 0.75

        if pvout is not None:
            pvout_daily = pvout
            pvout_source = "global_solar_atlas"
        elif ghi is not None:
            pvout_daily = round((ghi / 365) * performance_ratio, 2)
            pvout_source = "estimated_from_ghi"
        else:
            pvout_daily = None
            pvout_source = "not_available"

        return {
            "lat": lat,
            "lon": lon,
            "atlas_url": atlas_url,

            # Best available daily PV output estimate.
            # Unit: kWh/kWp/day
            "pvout_daily": pvout_daily,
            "pvout_source": pvout_source,

            # Raw solar resource values from Global Solar Atlas.
            # GHI, DNI, DIF are usually annual kWh/m²/year.
            "ghi_annual": ghi,
            "ghi_daily": ghi_daily,
            "dni_annual": dni,
            "dif_annual": dif,
            "gti_annual": gti,

            # Other site indicators.
            "optimal_tilt_angle": opta,
            "temperature": temp,
        }

    def create_solar_estimate(monthly_bill, kwh_usage, effective_rate_per_kwh, solar_data):
        """
        Creates the Sunspark MVP solar estimate.

        Formula:
        monthlyProductionPerKwp = pvOutputDaily * 30
        targetOffsetPercent = 0.70
        targetKwhOffset = monthlyKwhUsage * targetOffsetPercent
        recommendedSystemSizeKwp = targetKwhOffset / monthlyProductionPerKwp
        estimatedMonthlySolarKwh = recommendedSystemSizeKwp * monthlyProductionPerKwp
        estimatedMonthlySavings = estimatedMonthlySolarKwh * effectiveRatePerKwh
        estimatedAnnualSavings = estimatedMonthlySavings * 12
        estimatedNewBill = monthlyBill - estimatedMonthlySavings
        estimatedInstallCost = recommendedSystemSizeKwp * costPerKwp
        paybackYears = estimatedInstallCost / estimatedAnnualSavings
        monthlyCo2ReductionKg = estimatedMonthlySolarKwh * gridEmissionFactor
        annualCo2ReductionTons = (monthlyCo2ReductionKg * 12) / 1000
        """

        pv_output_daily = solar_data.get("pvout_daily")

        target_offset_percent = 0.70

        # MVP assumption.
        # You can adjust this later based on real PH solar developer pricing.
        cost_per_kwp = 60000

        # Approximate Philippines grid emission factor.
        # Unit: kg CO2 per kWh.
        # You can refine this later using official DOE/Grid data.
        grid_emission_factor = 0.70

        if (
            monthly_bill is None
            or kwh_usage is None
            or effective_rate_per_kwh is None
            or pv_output_daily is None
            or kwh_usage <= 0
            or effective_rate_per_kwh <= 0
            or pv_output_daily <= 0
        ):
            return {
                "target_offset_percent": target_offset_percent,
                "monthly_production_per_kwp": None,
                "target_kwh_offset": None,
                "recommended_system_size_kwp": None,
                "estimated_monthly_solar_kwh": None,
                "estimated_monthly_production_kwh": None,
                "estimated_monthly_savings": None,
                "estimated_annual_savings": None,
                "estimated_new_bill": None,
                "estimated_install_cost": None,
                "cost_per_kwp": cost_per_kwp,
                "payback_years": None,
                "grid_emission_factor": grid_emission_factor,
                "monthly_co2_reduction_kg": None,
                "annual_co2_reduction_tons": None,
                "coverage_percentage": None,
            }

        monthly_production_per_kwp = pv_output_daily * 30

        target_kwh_offset = kwh_usage * target_offset_percent

        recommended_system_size_kwp = target_kwh_offset / monthly_production_per_kwp

        estimated_monthly_solar_kwh = (
            recommended_system_size_kwp * monthly_production_per_kwp
        )

        estimated_monthly_savings = (
            estimated_monthly_solar_kwh * effective_rate_per_kwh
        )

        estimated_annual_savings = estimated_monthly_savings * 12

        estimated_new_bill = monthly_bill - estimated_monthly_savings

        estimated_install_cost = recommended_system_size_kwp * cost_per_kwp

        payback_years = (
            estimated_install_cost / estimated_annual_savings
            if estimated_annual_savings > 0
            else None
        )

        monthly_co2_reduction_kg = (
            estimated_monthly_solar_kwh * grid_emission_factor
        )

        annual_co2_reduction_tons = (
            monthly_co2_reduction_kg * 12
        ) / 1000

        coverage_percentage = target_offset_percent * 100

        return {
            "target_offset_percent": round(target_offset_percent, 2),
            "monthly_production_per_kwp": round(monthly_production_per_kwp, 2),
            "target_kwh_offset": round(target_kwh_offset, 2),
            "recommended_system_size_kwp": round(recommended_system_size_kwp, 2),
            "estimated_monthly_solar_kwh": round(estimated_monthly_solar_kwh, 2),

            # Kept for frontend compatibility with your current app.
            "estimated_monthly_production_kwh": round(estimated_monthly_solar_kwh, 2),

            "estimated_monthly_savings": round(estimated_monthly_savings, 2),
            "estimated_annual_savings": round(estimated_annual_savings, 2),
            "estimated_new_bill": round(estimated_new_bill, 2),
            "estimated_install_cost": round(estimated_install_cost, 2),
            "cost_per_kwp": cost_per_kwp,
            "payback_years": round(payback_years, 1) if payback_years is not None else None,
            "grid_emission_factor": grid_emission_factor,
            "monthly_co2_reduction_kg": round(monthly_co2_reduction_kg, 2),
            "annual_co2_reduction_tons": round(annual_co2_reduction_tons, 2),
            "coverage_percentage": round(coverage_percentage, 2),
        }

    @web_app.get("/")
    def health_check():
        return {
            "status": "ok",
            "message": "Sunspark OCR backend is running",
        }

    @web_app.post("/ocr")
    async def extract_text(file: UploadFile = File(...)):
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            ocr_text = pytesseract.image_to_string(image).strip()

            return {
                "success": True,
                "text": ocr_text,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
        
    @web_app.get("/reverse-geocode")
    async def reverse_geocode(lat: float, lon: float):
        try:
            location_data = get_location_name_from_nominatim(lat, lon)

            return {
                "success": True,
                "location": location_data,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @web_app.get("/solar-data")
    async def solar_data(lat: float, lon: float):
        """
        Simple test endpoint for checking solar data without uploading an image.
        Example:
        /solar-data?lat=30.317582&lon=44.851704
        """
        try:
            solar = get_global_solar_atlas_data(lat, lon)

            return {
                "success": True,
                "solar": solar,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    @web_app.post("/extract-bill-from-image")
    async def extract_bill_from_image(
        file: UploadFile = File(...),
        lat: float = Form(...),
        lon: float = Form(...),
    ):
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            ocr_text = pytesseract.image_to_string(image).strip()

            extracted_bill_data = extract_bill_fields_with_llm(ocr_text)
            solar_data = get_global_solar_atlas_data(lat, lon)
            location_data = get_location_name_from_nominatim(lat, lon)

            estimate = create_solar_estimate(
                monthly_bill=extracted_bill_data.get("monthly_bill"),
                kwh_usage=extracted_bill_data.get("kwh_usage"),
                effective_rate_per_kwh=extracted_bill_data.get("effective_rate_per_kwh"),
                solar_data=solar_data,
            )

            return {
                "success": True,
                **extracted_bill_data,
                "location": location_data,
                "solar": solar_data,
                "estimate": estimate,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    return web_app