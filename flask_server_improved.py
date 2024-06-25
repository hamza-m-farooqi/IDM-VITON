import io
from typing import List
from PIL import Image
from flask import Flask, request, jsonify, send_file
from idm_viton_abstraction import idm_viton_abstraction

app = Flask(__name__)

idm_viton_instances: "List[idm_viton_abstraction]" = []


@app.route("/try-on", methods=["POST"])
def api_tryon():
    try:

        # Image files are expected to be passed as part of form data
        if not len(idm_viton_instances) > 0:
            idm_viton_instances.append(idm_viton_abstraction())

        garm_img = Image.open(request.files["garm_img"])
        human_img = Image.open(request.files["human_img"])

        # Other parameters passed as part of form data
        garment_description = request.form["description"]
        garment_type = request.form["garment_type"]
        if garment_type not in ["dresses", "upper_body", "lower_body"]:
            raise Exception(
                "Invalid Garment Type. Available Options are dresses, upper_body, lower_body"
            )
        is_checked = request.form["is_checked"].lower() == "true"
        denoise_steps = 30
        seed = 42

        # Example dictionary for your process
        dict_inputs = {
            "background": human_img,
            "layers": [human_img],  # Depending on your application logic
            "composite": None,
        }

        # Process the image through your function
        output_image, mask_image = idm_viton_instances[0].start_tryon(
            dict_inputs,
            garm_img,
            garment_description,
            is_checked,
            denoise_steps,
            seed,
            garment_type,
        )

        # Save the output image to a byte stream and send it as a file
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        mask_byte_arr = io.BytesIO()
        mask_image.save(mask_byte_arr, format="PNG")
        mask_byte_arr.seek(0)

        # Sending back as multipart file response
        response = send_file(
            img_byte_arr, download_name="output.png", mimetype="image/png"
        )
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # if not len(idm_viton_instances) > 0:
    #     idm_viton_instances.append(idm_viton_abstraction())
    app.run(debug=True, host="0.0.0.0", port=5000)
