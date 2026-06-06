import modal

app = modal.App("sunspark")

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
def SunsparkBackend():
    from fastapi import FastAPI, File, UploadFile, Form, Body
    from fastapi.middleware.cors import CORSMiddleware
    from PIL import Image
    from google import genai
    from google.genai import types
    import pytesseract
    import io
    import json
    import math
    import os
    import re
    import requests

    web_app = FastAPI()

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Helpers

    def clean_number(value):
        """Strips non-numeric characters and returns a float, or None."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = re.sub(r"[^0-9.]", "", str(value))
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    def safe_round(value, decimals):
        """Returns rounded value, or None if value is None."""
        return round(value, decimals) if value is not None else None

    def ceil_to_increment(value, increment):
        """Rounds value up to the next increment, or None if value is None."""
        return math.ceil(value / increment) * increment if value is not None else None

    def get_nested(data, *keys):
        """Safely traverses nested dicts. Returns None if any key is missing."""
        for key in keys:
            if not isinstance(data, dict):
                return None
            data = data.get(key)
        return data

    def normalize_solar_value(value):
        """
        Global Solar Atlas values can be plain numbers or dicts.
        Normalizes both shapes into a float.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("value", "annual", "avg", "data"):
                if key in value:
                    return clean_number(value[key])
        return clean_number(value)

    # Bill Extraction

    def extract_bill_fields_with_llm(ocr_text: str):
        """
        Uses Gemini to extract structured fields from raw OCR text of an
        electricity bill. Derives effective rate and average monthly bill
        from the extracted values.
        """
        prompt = f"""
You are extracting structured data from OCR text of an electricity bill.

Extract only these fields:
1. monthly_bill - the total amount due or current monthly bill amount
2. kwh_usage - the electricity consumption for the current billing period in kWh
3. avg_monthly_kwh - the average monthly consumption in kWh, usually found in a
   "typical consumption", "average monthly consumption", or similar summary section.
   This is often labeled as "Ave. monthly consumption (last 12 months)" or similar.
4. rate_per_kwh - the electricity rate per kWh only if clearly shown in the bill
5. customer_type - the customer classification shown on the bill, such as
   "Residential", "Commercial", or "Industrial". Return null if not found.

Rules:
- Return null if a field is missing or uncertain. Do not guess.
- monthly_bill, kwh_usage, avg_monthly_kwh, rate_per_kwh: numbers only, no symbols.
- avg_monthly_kwh: return null if not explicitly shown on the bill.
- customer_type: normalize capitalization (e.g. "Residential"). Return null if not found.
- For monthly_bill, prefer: total amount due, amount due, current amount due, total current charges.
- For kwh_usage, prefer: total kWh used, energy consumption, consumption, present month usage.
- For rate_per_kwh, only extract a rate directly shown (generation charge, supply rate,
  energy charge rate). Do not compute it.

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
                        "monthly_bill": {"type": "number", "nullable": True},
                        "kwh_usage": {"type": "number", "nullable": True},
                        "avg_monthly_kwh": {"type": "number", "nullable": True},
                        "rate_per_kwh": {"type": "number", "nullable": True},
                        "customer_type": {"type": "string", "nullable": True},
                    },
                    "required": [
                        "monthly_bill", "kwh_usage", "avg_monthly_kwh",
                        "rate_per_kwh", "customer_type",
                    ],
                },
            ),
        )

        extracted = json.loads(response.text)

        monthly_bill = clean_number(extracted.get("monthly_bill"))
        kwh_usage = clean_number(extracted.get("kwh_usage"))
        avg_monthly_kwh = clean_number(extracted.get("avg_monthly_kwh"))
        rate_per_kwh_found_on_bill = clean_number(extracted.get("rate_per_kwh"))
        customer_type = extracted.get("customer_type") or None

        # Prefer avg_monthly_kwh for effective rate — more stable than single-period usage.
        kwh_for_rate = avg_monthly_kwh if avg_monthly_kwh is not None else kwh_usage
        effective_rate_per_kwh = (
            round(monthly_bill / kwh_for_rate, 4)
            if monthly_bill and kwh_for_rate and kwh_for_rate > 0
            else None
        )

        avg_monthly_bill = (
            round(avg_monthly_kwh * effective_rate_per_kwh, 2)
            if avg_monthly_kwh is not None and effective_rate_per_kwh is not None
            else None
        )

        return {
            "monthly_bill": monthly_bill,
            "kwh_usage": kwh_usage,
            "avg_monthly_kwh": avg_monthly_kwh,
            "avg_monthly_bill": avg_monthly_bill,
            "rate_per_kwh_found_on_bill": rate_per_kwh_found_on_bill,
            "effective_rate_per_kwh": effective_rate_per_kwh,
            "customer_type": customer_type,
        }

    # Location

    def get_location_from_nominatim(lat: float, lon: float):
        """
        Reverse geocodes coordinates using OpenStreetMap Nominatim.
        zoom=13 returns village/suburb-level data for barangay resolution in PH.
        """
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 13, "addressdetails": 1},
            timeout=20,
            headers={"User-Agent": "Sunspark/0.1 (ianjure.data@gmail.com)", "Accept": "application/json"},
        )
        response.raise_for_status()

        address = response.json().get("address", {})

        return {
            "barangay": (
                address.get("village")
                or address.get("suburb")
                or address.get("neighbourhood")
                or address.get("quarter")
                or address.get("hamlet")
            ),
            "city_or_municipality": (
                address.get("city")
                or address.get("town")
                or address.get("municipality")
                or address.get("county")
            ),
            "province": address.get("state") or address.get("province"),
        }

    # Solar Data

    def get_solar_data(lat: float, lon: float):
        """Fetches annual solar resource data from the Global Solar Atlas API."""
        response = requests.get(
            "https://api.globalsolaratlas.info/data/lta",
            params={"loc": f"{lat},{lon}"},
            timeout=20,
        )
        response.raise_for_status()

        annual = get_nested(response.json(), "annual", "data") or {}
        gv = lambda key: normalize_solar_value(annual.get(key))  # shorthand

        pvout = gv("PVOUT_csi")
        ghi = gv("GHI")

        return {
            "lat": lat,
            "lon": lon,
            "atlas_url": f"https://globalsolaratlas.info/map?c={lat},{lon},11&s={lat},{lon}",
            "pvout_daily": safe_round(pvout / 365, 4) if pvout else None,
            "pvout_source": "Global Solar Atlas",
            "ghi_annual": ghi,
            "ghi_daily": safe_round(ghi / 365, 4) if ghi else None,
            "dni_annual": gv("DNI"),
            "dif_annual": gv("DIF"),
            "gti_annual": gv("GTI_opta"),
            "optimal_tilt_angle": gv("OPTA"),
            "temperature": gv("TEMP"),
        }

    # Solar Estimate

    def create_solar_estimate(monthly_bill, avg_monthly_kwh, effective_rate_per_kwh, solar_data):
        """
        Sizes a solar system to offset 70% of average monthly consumption.

        Uses avg_monthly_kwh (not the single-period bill) as the baseline to
        avoid over/under-sizing from seasonal variation.
        """
        pv_output_daily = solar_data.get("pvout_daily")

        # MVP constants — adjust as real PH market/grid data becomes available.
        TARGET_OFFSET = 0.70
        COST_PER_KWP = 60_000       # PHP; based on typical PH installer pricing
        GRID_EMISSION = 0.70        # kg CO₂ per kWh; approximate PH grid factor

        has_required = all([
            monthly_bill, avg_monthly_kwh, effective_rate_per_kwh, pv_output_daily,
            avg_monthly_kwh > 0, effective_rate_per_kwh > 0, pv_output_daily > 0,
        ])

        null_estimate = {
            "target_offset_percent": TARGET_OFFSET,
            "monthly_production_per_kwp": None,
            "target_kwh_offset": None,
            "recommended_system_size_kwp": None,
            "estimated_monthly_solar_kwh": None,
            "estimated_monthly_production_kwh": None,
            "estimated_monthly_savings": None,
            "estimated_annual_savings": None,
            "estimated_new_bill": None,
            "estimated_install_cost": None,
            "cost_per_kwp": COST_PER_KWP,
            "payback_years": None,
            "grid_emission_factor": GRID_EMISSION,
            "monthly_co2_reduction_kg": None,
            "annual_co2_reduction_tons": None,
            "coverage_percentage": None,
        }

        if not has_required:
            return null_estimate

        monthly_prod_per_kwp = pv_output_daily * 30
        target_kwh_offset = avg_monthly_kwh * TARGET_OFFSET
        system_size_kwp = target_kwh_offset / monthly_prod_per_kwp
        monthly_solar_kwh = system_size_kwp * monthly_prod_per_kwp
        monthly_savings = monthly_solar_kwh * effective_rate_per_kwh
        annual_savings = monthly_savings * 12
        install_cost = system_size_kwp * COST_PER_KWP
        monthly_co2_kg = monthly_solar_kwh * GRID_EMISSION

        return {
            "target_offset_percent": round(TARGET_OFFSET, 2),
            "monthly_production_per_kwp": round(monthly_prod_per_kwp, 2),
            "target_kwh_offset": round(target_kwh_offset, 2),
            "recommended_system_size_kwp": ceil_to_increment(system_size_kwp, 0.5),
            "estimated_monthly_solar_kwh": round(monthly_solar_kwh, 2),
            "estimated_monthly_production_kwh": round(monthly_solar_kwh, 2),
            "estimated_monthly_savings": ceil_to_increment(monthly_savings, 100),
            "estimated_annual_savings": round(annual_savings, 2),
            "estimated_new_bill": round(monthly_bill - monthly_savings, 2),
            "estimated_install_cost": ceil_to_increment(install_cost, 100_000),
            "cost_per_kwp": COST_PER_KWP,
            "payback_years": math.ceil(install_cost / annual_savings) if annual_savings > 0 else None,
            "grid_emission_factor": GRID_EMISSION,
            "monthly_co2_reduction_kg": round(monthly_co2_kg, 2),
            "annual_co2_reduction_tons": round((monthly_co2_kg * 12) / 1000, 2),
            "coverage_percentage": round(TARGET_OFFSET * 100, 2),
        }
    
    def build_bill_fields_from_values(
        monthly_bill,
        kwh_usage,
        avg_monthly_kwh,
        rate_per_kwh_found_on_bill=None,
        effective_rate_per_kwh=None,
        customer_type=None,
    ):
        monthly_bill = clean_number(monthly_bill)
        kwh_usage = clean_number(kwh_usage)
        avg_monthly_kwh = clean_number(avg_monthly_kwh)
        rate_per_kwh_found_on_bill = clean_number(rate_per_kwh_found_on_bill)
        effective_rate_per_kwh = clean_number(effective_rate_per_kwh)

        kwh_for_rate = avg_monthly_kwh if avg_monthly_kwh is not None else kwh_usage

        if effective_rate_per_kwh is None:
            effective_rate_per_kwh = (
                round(monthly_bill / kwh_for_rate, 4)
                if monthly_bill and kwh_for_rate and kwh_for_rate > 0
                else None
            )

        avg_monthly_bill = (
            round(avg_monthly_kwh * effective_rate_per_kwh, 2)
            if avg_monthly_kwh is not None and effective_rate_per_kwh is not None
            else None
        )

        return {
            "monthly_bill": monthly_bill,
            "kwh_usage": kwh_usage,
            "avg_monthly_kwh": avg_monthly_kwh,
            "avg_monthly_bill": avg_monthly_bill,
            "rate_per_kwh_found_on_bill": rate_per_kwh_found_on_bill,
            "effective_rate_per_kwh": effective_rate_per_kwh,
            "customer_type": customer_type or None,
        }

    # Readiness Score

    def calculate_readiness_score(estimate, solar_data, assessment_answers):
        """
        Scores solar readiness 0–100 based on solar resource, system fit,
        and user assessment answers. Stored with user data for future analysis.
        """
        score = 60

        coverage = (estimate or {}).get("coverage_percentage") or 0
        pvout_daily = (solar_data or {}).get("pvout_daily") or 0
        sunlight = (assessment_answers or {}).get("sunlight")
        roof_space = (assessment_answers or {}).get("roof_space")

        if coverage >= 70:  score += 10
        if pvout_daily >= 4: score += 10

        if sunlight == "Mostly sunny":     score += 10
        elif sunlight == "Partially shaded": score += 5
        elif sunlight == "Heavily shaded":   score -= 5

        if roof_space == "Large":  score += 10
        elif roof_space == "Medium": score += 5
        elif roof_space == "Small":  score -= 5

        return max(0, min(100, score))

    def get_readiness_label(score: int) -> str:
        if score >= 80: return "GREAT FIT FOR SOLAR"
        if score >= 60: return "GOOD FIT FOR SOLAR"
        return "NEEDS MORE REVIEW"

    # Routes

    @web_app.get("/")
    def health_check():
        return {"status": "ok", "message": "Sunspark backend is running"}

    @web_app.post("/ocr")
    async def ocr(file: UploadFile = File(...)):
        """Returns raw OCR text from an uploaded bill image. Useful for debugging."""
        try:
            image = Image.open(io.BytesIO(await file.read()))
            return {"success": True, "text": pytesseract.image_to_string(image).strip()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @web_app.get("/reverse-geocode")
    async def reverse_geocode(lat: float, lon: float):
        try:
            return {"success": True, "location": get_location_from_nominatim(lat, lon)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @web_app.get("/solar-data")
    async def solar_data(lat: float, lon: float):
        """Test endpoint for checking solar data without a bill upload."""
        try:
            return {"success": True, "solar": get_solar_data(lat, lon)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @web_app.post("/extract-data-from-image")
    async def extract_data_from_image(
        file: UploadFile = File(...),
        lat: float = Form(...),
        lon: float = Form(...),
        home_ownership: str = Form(None),
        sunlight: str = Form(None),
        roof_space: str = Form(None),
        payment_preference: str = Form(None),
        installation_timeline: str = Form(None),
        user_name: str = Form(None),
    ):
        try:
            image = Image.open(io.BytesIO(await file.read()))
            ocr_text = pytesseract.image_to_string(image).strip()

            bill = extract_bill_fields_with_llm(ocr_text)
            solar = get_solar_data(lat, lon)
            location = get_location_from_nominatim(lat, lon)

            assessment_answers = {
                "home_ownership": home_ownership,
                "sunlight": sunlight,
                "roof_space": roof_space,
                "payment_preference": payment_preference,
                "installation_timeline": installation_timeline,
            }

            estimate = create_solar_estimate(
                monthly_bill=bill.get("monthly_bill"),
                avg_monthly_kwh=bill.get("avg_monthly_kwh"),
                effective_rate_per_kwh=bill.get("effective_rate_per_kwh"),
                solar_data=solar,
            )

            readiness_score = calculate_readiness_score(estimate, solar, assessment_answers)

            return {
                "success": True,
                "user_name": user_name,
                **bill,
                "location": location,
                "solar": solar,
                "estimate": estimate,
                "assessment_answers": assessment_answers,
                "readiness_score": readiness_score,
                "readiness_label": get_readiness_label(readiness_score),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @web_app.post("/recompute-estimate")
    async def recompute_estimate(payload: dict = Body(...)):
        try:
            bill = build_bill_fields_from_values(
                monthly_bill=payload.get("monthly_bill"),
                kwh_usage=payload.get("kwh_usage"),
                avg_monthly_kwh=payload.get("avg_monthly_kwh"),
                rate_per_kwh_found_on_bill=payload.get("rate_per_kwh_found_on_bill")
                    or payload.get("rate_per_kwh"),
                effective_rate_per_kwh=payload.get("effective_rate_per_kwh"),
                customer_type=payload.get("customer_type"),
            )

            solar = payload.get("solar") or payload.get("solar_data")
            lat = clean_number(payload.get("lat"))
            lon = clean_number(payload.get("lon"))

            if not isinstance(solar, dict):
                if lat is None or lon is None:
                    return {
                        "success": False,
                        "error": "Either solar data or both lat and lon are required.",
                    }
                solar = get_solar_data(lat, lon)

            assessment_answers = payload.get("assessment_answers") or {
                "home_ownership": payload.get("home_ownership"),
                "sunlight": payload.get("sunlight"),
                "roof_space": payload.get("roof_space"),
                "payment_preference": payload.get("payment_preference"),
                "installation_timeline": payload.get("installation_timeline"),
            }

            estimate = create_solar_estimate(
                monthly_bill=bill.get("monthly_bill"),
                avg_monthly_kwh=bill.get("avg_monthly_kwh"),
                effective_rate_per_kwh=bill.get("effective_rate_per_kwh"),
                solar_data=solar,
            )

            readiness_score = calculate_readiness_score(
                estimate,
                solar,
                assessment_answers,
            )

            response = {
                "success": True,
                **bill,
                "solar": solar,
                "estimate": estimate,
                "assessment_answers": assessment_answers,
                "readiness_score": readiness_score,
                "readiness_label": get_readiness_label(readiness_score),
            }

            if payload.get("location") is not None:
                response["location"] = payload.get("location")

            return response

        except Exception as e:
            return {"success": False, "error": str(e)}

    return web_app
