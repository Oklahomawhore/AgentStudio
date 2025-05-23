PREPRODUCTION_PROMPTS = {
    "Concept Planning": """
    You are a truthful video creater specialized at recreating experiences using video given text scripts, your job is to help the audience understand the story/content based on video.

    Given a short story, please write a video concept based on the input story, to help the production team understand what is the big picture in the video.
    Your concept planning should include instruction about the main idea, the involving characters and story progression.

    END GOAL: the audience should be able to recreate the experience in their head by only looking at the video.
    """,
    "Script Writing": """
    Write the plays based on the scripts in the previous scripting step. Each play should have clear boundary and extremely simple motion and setup, 
    Here a ''play`` is not what is common in operas or in theatre or in movies, a play is just a fragment of the development of the current situation, 
    which is coherent with both the preceding and the following act to make the whole story both logically and visually consistent.
    """,
    "Casting Design": """
    Select and hire actors. For each character in the script, create a detailed casting requirement sheet, including descriptions of age, gender, nationality, time period, skin color, appearance, clothing, and other relevant details.
    Particularly, ensure that the character's physical appearance is described in great detail. This character may be a fictional one that does not exist in reality.
    """,
    "Casting Extraction" : """
    Extract the casting requirement sheets from the casting and equipment sections and generate a JSON object for further processing. For example:
    ```
    {
        "Song Gong": "Around 50 years old, thin with a scholarly aura, fair-skinned, mid-Qing Dynasty, China. Dressed in slightly official attire, wearing a mang robe, a seventh-rank hat with a peacock feather, a wide belt around the waist, and black cloth shoes.",
        "Zhang Xiucai": "Around 40 years old, slightly chubby, short, fair-skinned, mid-Qing Dynasty, China. Wearing simple Qing Dynasty civilian clothing, a split-front long robe with sleeves, a wide belt around the waist, a black round hat, and black cloth shoes.",
        "King Yanluo": "Unknown age, tall, serious expression, dark complexion, mid-Qing Dynasty, China. Wearing imperial attire, adorned with a Tongtian crown, including a beam crown, red robe, red skirt, large belt, leather belt, ribbon, white socks, and black wooden clogs.",
        "Guan Gong": "Unknown age, tall, serious expression, reddish complexion, Three Kingdoms period, China. Wearing battle armor, a red helmet, and wielding a Green Dragon Crescent Blade."
    }
    ```
    """,
    "Detailed Storyboarding": """
    Extracting Shot Descriptions from a Storyboard

    Generate a JSON object from the storyboard, where each shot is structured for further processing.

    Note:
    	•	Each video segment is fixed at 5 seconds in length, but the music track can span multiple shots.
        •   Design only number of parts of music based on setting, scenes that is in the same setting shares the *IDENTICAL* music description.
    	•	Dialogue must use "\\" notation instead of regular quotation marks.
    	•	If the scene does not contain a frontal human face, use "t2v" (text-to-video).
    	•	If the scene contains a frontal human face, use "i2v" (image-to-video).
    	•	If the user does not provide an image, the first scene cannot be "i2v".
        •   *ALL* chracters in the story should be marked with "<#name#>" notation. e.g. "<#Holmes#>".

    For example, using a well-known English-language setting like Victorian London, the JSON structure could look like this:
    MAKE SURE TO GENERATE THE WHOLE JSON OBJECT NON-STOP!!!

    {
        "1": {
            "scene": "Medium shot",
            "style": "Realistic",
            "act" : "Setup",
            "content": "Victorian-era London. A dimly lit study filled with bookshelves and antique furniture. Nighttime, a single candle flickers on a wooden desk. <#Mr. Holmes#> sits in a high-backed chair, eyes closed, fingers pressed together in deep thought. Outside, rain patters against the window, casting ghostly reflections.",
            "duration": 5,
            "motion": "Slow push-in",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "None",
            "type": "t2v"
        },
        "2": {
            "scene": "Close-up",
            "style": "Realistic",
            "act" : "Setup",
            "content": "The camera focuses on <#Holmes#> as he suddenly opens his piercing eyes. His gaze sharpens, as if sensing something unseen. A clock in the corner strikes midnight. The sound of a carriage stopping outside echoes through the quiet room.",
            "duration": 5,
            "motion": "Subtle shake",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "None",
            "type": "i2v"
        },
        "3": {
            "scene": "Medium shot",
            "style": "Realistic",
            "act" : "Setup",
            "content": "A knock on the door. <#Dr. Watson#>, dressed in his evening robe, enters hurriedly, concern etched on his face. He steps into the dim light and speaks urgently:",
            "duration": 5,
            "motion": "Gentle push-in",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "<#Watson#>: \"There’s been another murder, Holmes. The police need you immediately.\"",
            "type": "i2v"
        },
        "4": {
            "scene": "Wide shot",
            "style": "Mystical",
            "act" : "Setup",
            "content": "Fog swirls through the cobbled streets of Victorian London. A hansom cab speeds through the misty night, carrying <#Holmes#> and <#Watson#> toward the crime scene. Gas lamps flicker, casting eerie shadows on the damp pavement.",
            "duration": 5,
            "motion": "Tracking shot from behind",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "None",
            "type": "t2v"
        },
        "5": {
            "scene": "Medium shot",
            "style": "Mystical",
            "act" : "Setup",
            "content": "<#Holmes#> and <#Watson#> arrive at a grand but decaying mansion. The heavy wooden doors creak open, revealing a dimly lit hallway with faded wallpaper and dust-covered chandeliers. The air is thick with unease.",
            "duration": 5,
            "motion": "Slow panning shot to establish setting",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "None",
            "type": "t2v"
        },
        "6": {
            "scene": "Wide shot",
            "style": "Ethereal",
            "act" : "Setup",
            "content": "Dawn breaks over London. Big Ben stands tall against the golden sky. The streets are now bustling, oblivious to the darkness that lurks beneath the city's surface. <#Holmes#> stands alone on a bridge, staring into the distance, lost in thought.",
            "duration": 5,
            "motion": "Slow pan across the skyline",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "None",
            "type": "t2v"
        },
        "7": {
            "scene": "Close-up",
            "style": "Dramatic",
            "act" : "Setup",
            "content": "<#Watson#> places a reassuring hand on <#Holmes#>'s shoulder. His expression is full of concern.",
            "duration": 5,
            "motion": "Gentle push-in",
            "music": "A soft violin melody, carrying an air of mystery, accompanied by subtle piano chord, underscoring tension",
            "dialogue": "<#Watson#>: \"You can’t solve everything alone, my friend.\"",
            "type": "i2v"
        },
        "8": {
            "scene": "Extreme close-up",
            "style": "Thriller",
            "act" : "Confrontation",
            "content": "A gloved hand slips a blood-stained letter into a wooden drawer. The camera lingers on the crimson seal, marked with an unfamiliar symbol.",
            "duration": 5,
            "motion": "Still frame, then a sudden zoom-in",
            "music": "A sharp piano note strikes, signaling danger, followed by a low rumble of thunder, underscoring suspense",
            "dialogue": "None",
            "type": "t2v"
        },
        "9": {
            "scene": "Medium shot",
            "style": "Thriller",
            "act" : "Confrontation",
            "content": "Back in Baker Street, <#Holmes#> studies the letter under the flickering gaslight. He slowly turns to <#Watson#> with a grave expression.",
            "duration": 5,
            "motion": "Steady push-in",
            "music": "A sharp piano note strikes, signaling danger, followed by a low rumble of thunder, underscoring suspense",
            "dialogue": "<#Holmes#>: \"This… changes everything.\"",
            "type": "i2v"
        },
        "10": {
            "scene": "Wide shot",
            "style": "Epic",
            "act" : "Confrontation",
            "content": "On the rooftops of London, a shadowy figure watches from above. His silhouette blends with the night as he vanishes into the mist.",
            "duration": 5,
            "motion": "Slow tracking shot, followed by a fade-out",
            "music": "A sharp piano note strikes, signaling danger, followed by a low rumble of thunder, underscoring suspense",
            "dialogue": "None",
            "type": "t2v"
        }
    }
    """,
}