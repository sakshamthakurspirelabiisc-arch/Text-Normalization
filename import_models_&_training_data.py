import gdown
import os

# =====================================================
# ROOT DIRECTORY
# =====================================================

ROOT_DIR = r"D:\TN_Models_"  # CHANGE THIS

# =====================================================
# DOWNLOAD LINKS
# =====================================================

folders = {

    # -------------------------------------------------
    # DATA
    # -------------------------------------------------

    "data": [
        "https://drive.google.com/file/d/1iX5tlGzNSKZtK2Owg6oE_2MVpXF5meUV/view?usp=drive_link",
        "https://drive.google.com/file/d/1dL-XS1XSi0Ve6ANp6P6dj8D_bBEYM4b_/view?usp=drive_link",
        "https://drive.google.com/file/d/1aSAlqILvrP8RRyTG1QP8_NMoIrBq8drO/view?usp=drive_link",
    ],

    # -------------------------------------------------
    # HINTS IB
    # -------------------------------------------------

    "hints_ib/ner/hi": [
        "https://drive.google.com/file/d/1TVH59QZBsFv0b4Y0MeT1fA0jUoP7Icrv/view?usp=drive_link",
    ],

    "hints_ib/ner/kan": [
        "https://drive.google.com/file/d/1GJ5_2OBhQPWxmyPCFzHIyRPsKzr0vm3c/view?usp=drive_link",
    ],

    "hints_ib/decoder/hi": [
        "https://drive.google.com/file/d/1nMgs9S8pG6ywZ1cDn8gzcIAEh-7zwY7l/view?usp=drive_link",
    ],

    "hints_ib/decoder/kan": [
        "https://drive.google.com/file/d/1ODIb_AhpCDHDZWiH-bFJt9abBvSr7rzx/view?usp=drive_link",
    ],

    # -------------------------------------------------
    # LM SENTENCE SELECTOR
    # -------------------------------------------------
# https://drive.google.com/file/d/1SkziJHkF1YQgh48CR42BHBwrE6FaYS-c/view?usp=drive_link
    "lm_sentence_selector/hi": [
        "https://drive.google.com/file/d/1-ACT0hTsw214iPXfxcE6JOPg4N-c5jvs/view?usp=drive_link",
        "https://drive.google.com/file/d/1SkziJHkF1YQgh48CR42BHBwrE6FaYS-c/view?usp=drive_link",
    ],

    "lm_sentence_selector/kan": [
        "https://drive.google.com/file/d/1DNcCdcd9A6mXqXTncLfjirxAI5FkYPPD/view?usp=drive_link",
    ],

    # -------------------------------------------------
    # LM SPAN SELECTOR
    # -------------------------------------------------

    "lm_span_selector/hi": [
        "https://drive.google.com/file/d/1kCT9q3MbbdjQqUw5UVEqy36M5GcbfnqO/view?usp=drive_link",
    ],

    "lm_span_selector/kan": [
        "https://drive.google.com/file/d/1d80w_RrRNtLiXwbOC0JO56wmOZxZediq/view?usp=drive_link",
    ],
}

# =====================================================
# CUSTOM FILE NAMES
# =====================================================

custom_names = {
    "data": [
        "train_hi.txt",
        "valid_hi.txt",
        "train_kan.txt",
    ]
}

# =====================================================
# DOWNLOAD
# =====================================================

for subfolder, links in folders.items():

    save_dir = os.path.join(ROOT_DIR, subfolder)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nDownloading to: {save_dir}")

    for idx, link in enumerate(links, start=1):

        if subfolder in custom_names:
            filename = custom_names[subfolder][idx - 1]
        else:
            filename = f"weight_{idx}.pt"

        output_file = os.path.join(save_dir, filename)

        try:
            gdown.download(
                url=link,
                output=output_file,
                fuzzy=True,
                quiet=False
            )

            print(f"✓ Saved: {output_file}")

        except Exception as e:
            print(f"✗ Failed: {link}")
            print(f"Error: {e}")

print("\nAll downloads complete.")