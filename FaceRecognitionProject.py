
import cv2
import numpy as np
import sqlite3
import os
import time
from datetime import datetime

# 1. CONFIGURATION DU PROJET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for folder in [DATABASE_DIR, DATASET_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

DB_PATH = os.path.join(DATABASE_DIR, "faces.db")
MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.yml")

FACE_SIZE = (200, 200)
ROI_SIZE = 260

# Seuil de reconnaissance à ajuster avec nos expériences.
# Plus la distance LBPH est petite, plus la correspondance est forte.
RECOGNITION_THRESHOLD = 65

# 2. CONNEXION A LA BASE DE DONNEES


def connect_database():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    connection.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            email TEXT,
            profession TEXT,
            birth_date TEXT,
            gender TEXT,
            country TEXT,
            city TEXT,
            organization TEXT,
            notes TEXT,
            primary_photo TEXT,
            created_at TEXT,
            last_seen TEXT
        )
    """)

    connection.commit()
    return connection


db = connect_database()

# 3. PREPARATION DES DOSSIERS DES PERSONNES


def person_folder(person_id):
    folder = os.path.join(DATASET_DIR, f"person_{person_id}")
    os.makedirs(folder, exist_ok=True)
    return folder


# 4. PREPARATION DES IMAGES


def prepare_face(frame):

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensionner toutes les images
    gray = cv2.resize(gray, FACE_SIZE)

    # Améliorer le contraste
    gray = cv2.equalizeHist(gray)

    return gray

# 5. COLLECTE GUIDEE

def collect_faces():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Impossible d'ouvrir la caméra.")
        return None, None

    window = "Collecte faciale"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    instructions = [
        ("Tourne ton visage a GAUCHE", "left"),
        ("Incline ton visage vers le BAS", "down"),
        ("Tourne ton visage a DROITE", "right"),
        ("Incline ton visage vers le HAUT", "up"),
        ("Regarde DEVANT toi et maintiens", "front")
    ]

    collected = {}

    try:

        for message, position in instructions:

            images = []
            start = time.time()

            while time.time() - start < 4:

                success, frame = camera.read()

                if not success:
                    raise RuntimeError("Lecture de caméra impossible.")

                height, width = frame.shape[:2]

                x1 = width // 2 - ROI_SIZE // 2
                y1 = height // 2 - ROI_SIZE // 2
                x2 = x1 + ROI_SIZE
                y2 = y1 + ROI_SIZE

                # Afficher le cadre de positionnement
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                remaining = max(
                    0,
                    4 - int(time.time() - start)
                )

                cv2.putText(
                    frame,
                    message,
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Temps restant : {remaining}s",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Place ton visage dans le cadre",
                    (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(window, frame)

                # Capture de trois images par position
                elapsed = time.time() - start

                if (
                    1.5 <= elapsed <= 3.5
                    and len(images) < 3
                ):
                    crop = frame[y1:y2, x1:x2]

                    if crop.shape[:2] == (ROI_SIZE, ROI_SIZE):
                        images.append(crop.copy())
                        time.sleep(0.35)

                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    return None, None

                if cv2.getWindowProperty(
                    window,
                    cv2.WND_PROP_VISIBLE
                ) < 1:
                    return None, None

            collected[position] = images

    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return collected, collected.get("front", [])


# 6. ENREGISTREMENT DES IMAGES

def save_images(person_id, collected):

    folder = person_folder(person_id)

    for position, images in collected.items():

        for index, image in enumerate(images):

            filename = f"{position}_{index + 1}.jpg"
            path = os.path.join(folder, filename)

            cv2.imwrite(path, image)

    # La photo principale sera une photo frontale
    front_images = collected.get("front", [])

    if front_images:

        primary_path = os.path.join(folder, "principal.jpg")

        cv2.imwrite(primary_path, front_images[0])

        return primary_path

    return None



# 7. FORMULAIRE D'ENREGISTREMENT

def ask_required_name():

    while True:

        name = input("\nNom complet (obligatoire) : ").strip()

        if name:
            return name

        print("Le nom complet est obligatoire. Réessaie.")


def ask_optional_information():

    print("\nInformations complémentaires")
    print("Tu peux appuyer sur ENTREE pour ignorer un champ.")

    information = {}

    information["phone"] = input("Téléphone : ").strip()
    information["email"] = input("E-mail : ").strip()
    information["profession"] = input("Profession : ").strip()
    information["birth_date"] = input("Date de naissance : ").strip()
    information["gender"] = input("Genre (facultatif) : ").strip()
    information["country"] = input("Pays : ").strip()
    information["city"] = input("Ville : ").strip()
    information["organization"] = input("Organisation : ").strip()
    information["notes"] = input("Informations supplémentaires : ").strip()

    return information


def register_person(collected):

    print("\n=== NOUVELLE PERSONNE ===")

    name = ask_required_name()

    information = ask_optional_information()

    now = datetime.now().isoformat(timespec="seconds")

    cursor = db.execute("""
        INSERT INTO persons (
            name, phone, email, profession,
            birth_date, gender, country, city,
            organization, notes, created_at, last_seen
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        information["phone"],
        information["email"],
        information["profession"],
        information["birth_date"],
        information["gender"],
        information["country"],
        information["city"],
        information["organization"],
        information["notes"],
        now,
        now
    ))

    person_id = cursor.lastrowid

    photo_path = save_images(person_id, collected)

    db.execute("""
        UPDATE persons
        SET primary_photo = ?
        WHERE id = ?
    """, (photo_path, person_id))

    db.commit()

    print(f"\n{ name } a été enregistré(e).")

    return person_id



# 8. CONSTRUCTION DU MODELE LBPH

def train_model():

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "Le module cv2.face est absent. "
            "Vérifie opencv-contrib-python."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    faces = []
    labels = []

    persons = db.execute(
        "SELECT id FROM persons"
    ).fetchall()

    for person in persons:

        person_id = person["id"]
        folder = person_folder(person_id)

        # Seules les images frontales servent à l'identification.
        # Les autres positions restent disponibles dans le dataset.

        for filename in os.listdir(folder):

            if not filename.startswith("front_"):
                continue

            if not filename.lower().endswith(".jpg"):
                continue

            path = os.path.join(folder, filename)

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, FACE_SIZE)
            image = cv2.equalizeHist(image)

            faces.append(image)
            labels.append(person_id)

    if not faces:
        return None

    recognizer.train(
        faces,
        np.array(labels, dtype=np.int32)
    )

    recognizer.save(MODEL_PATH)

    print(f"\nModèle entraîné avec {len(faces)} images.")

    return recognizer

# 9. RECONNAISSANCE FACIALE

def recognize_person(recognizer, front_images):

    if recognizer is None:
        return None

    predictions = []

    for image in front_images:

        face = prepare_face(image)

        person_id, confidence = recognizer.predict(face)

        if confidence <= RECOGNITION_THRESHOLD:
            predictions.append((person_id, confidence))

    if not predictions:
        return None

    # Retenir l'identité obtenant le plus de votes
    votes = {}

    for person_id, confidence in predictions:
        votes[person_id] = votes.get(person_id, 0) + 1

    best_id = max(votes, key=votes.get)

    # Exiger au moins deux correspondances concordantes
    if votes[best_id] < 2:
        return None

    return best_id

# 10. AFFICHAGE DE LA FICHE PERSON

def display_person_card(person_id):

    person = db.execute(
        "SELECT * FROM persons WHERE id = ?",
        (person_id,)
    ).fetchone()

    if person is None:
        return

    now = datetime.now().isoformat(timespec="seconds")

    db.execute(
        "UPDATE persons SET last_seen = ? WHERE id = ?",
        (now, person_id)
    )

    db.commit()

    # Créer une fiche visuelle
    card = np.full((500, 750, 3), 245, dtype=np.uint8)

    cv2.putText(
        card,
        "IDENTITE RECONNUE",
        (250, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 130, 0),
        2
    )

    photo = None

    if person["primary_photo"]:
        photo = cv2.imread(person["primary_photo"])

    if photo is not None:

        photo = cv2.resize(photo, (220, 260))

        card[100:360, 20:240] = photo

    else:

        cv2.putText(
            card,
            "Photo indisponible",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )

    details = [
        ("Nom", person["name"]),
        ("Telephone", person["phone"] or "Non renseigne"),
        ("E-mail", person["email"] or "Non renseigne"),
        ("Profession", person["profession"] or "Non renseignee"),
        ("Ville", person["city"] or "Non renseignee"),
        ("Pays", person["country"] or "Non renseigne"),
        ("Organisation", person["organization"] or "Non renseignee"),
        ("Premiere visite", person["created_at"] or ""),
        ("Derniere visite", person["last_seen"] or "")
    ]

    y = 120

    for label, value in details:

        cv2.putText(
            card,
            f"{label} : {value[:35]}",
            (265, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            1
        )

        y += 40

    cv2.putText(
        card,
        "Appuie sur Q pour fermer cette fiche",
        (160, 465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (80, 80, 80),
        1
    )

    cv2.imshow("Carte de la personne", card)

    while True:

        key = cv2.waitKey(50) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(
            "Carte de la personne",
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)



# 11. PROGRAMME PRINCIPALE

def main():

    print("=" * 45)
    print("SYSTEME DE RECONNAISSANCE FACIALE")
    print("=" * 45)

    # Charger ou entraîner le modèle à partir de la base
    recognizer = train_model()

    print("\nPlace ton visage devant la caméra.")
    print("La collecte va commencer.")

    collected, front_images = collect_faces()

    if collected is None:
        print("Collecte annulée.")
        return

    # Rechercher une identité connue
    person_id = recognize_person(
        recognizer,
        front_images
    )

    if person_id is not None:

        print("\nPersonne reconnue !")

        display_person_card(person_id)

    else:

        print("\nPersonne non reconnue.")

        # Enregistrer les renseignements et les images
        person_id = register_person(collected)

        # Réentraîner le modèle avec les nouvelles données
        recognizer = train_model()

        print("\nNouvelle personne enregistrée.")

        display_person_card(person_id)


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nProgramme interrompu.")

    except Exception as error:
        print("\nUne erreur est survenue :", error)

    finally:
        db.close()
        cv2.destroyAllWindows()
