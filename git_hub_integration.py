import os
from github import Github
from github import Auth
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env")

auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)

ALLOWED_EXTENSIONS = (".md", ".txt", ".py")
MAX_FILE_SIZE = 100_000  # 100 KB safety cap

def load_github_texts():
    texts = []
    user = g.get_user()

    for repo in user.get_repos():
        try:
            contents = repo.get_contents("")

            while contents:
                file = contents.pop(0)

                # Folder → drill down
                if file.type == "dir":
                    contents.extend(repo.get_contents(file.path))
                    continue

                # File filter
                if not file.name.endswith(ALLOWED_EXTENSIONS):
                    continue

                # Size guard
                if file.size > MAX_FILE_SIZE:
                    continue

                file_text = file.decoded_content.decode("utf-8", errors="ignore")

                texts.append(
                    f"Repository: {repo.name}\n"
                    f"File: {file.path}\n"
                    f"{file_text}"
                )

        except Exception as e:
            print(f"[WARN] Skipping repo {repo.name}: {e}")

    return texts
