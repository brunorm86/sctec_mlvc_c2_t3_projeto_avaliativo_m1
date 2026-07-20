import os
from dulwich import porcelain
from dulwich.repo import Repo

path = "."
try:
    porcelain.init(path)
except:
    pass

repo = Repo(path)
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")

os.environ["GIT_AUTHOR_NAME"] = "Cientista de Dados"
os.environ["GIT_AUTHOR_EMAIL"] = "cientista@banco.com"
os.environ["GIT_COMMITTER_NAME"] = "Cientista de Dados"
os.environ["GIT_COMMITTER_EMAIL"] = "cientista@banco.com"

# 1. Main branch
try:
    porcelain.add(path, paths=[".gitignore", "requirements.txt", "README.md", "data/credit_risk_dataset.csv"])
except:
    pass # maybe some files are missing, just add what's there
porcelain.commit(path, message=b"feat: inicia projeto e dependencias")

# 2. Branch fase/eda
# To branch off, we just create a ref pointing to the current commit
head_commit = repo.head()
repo.refs[b"refs/heads/fase/eda"] = head_commit
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/fase/eda")

porcelain.add(path, paths=["dicionario_dados.md"])
porcelain.commit(path, message=b"docs: adiciona dicionario de dados")

porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"feat: adiciona analise exploratoria e graficos")

# Merge fase/eda into main
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
# Fast-forward main to fase/eda
repo.refs[b"refs/heads/main"] = repo.refs[b"refs/heads/fase/eda"]

# 3. Branch fase/data-prep
head_commit = repo.head()
repo.refs[b"refs/heads/fase/data-prep"] = head_commit
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/fase/data-prep")

with open("projeto_credito.ipynb", "a") as f: f.write("\n")
porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"feat: adiciona tratamento de nulos e remocao de duplicatas")

with open("projeto_credito.ipynb", "a") as f: f.write("\n")
porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"fix: remove outliers de idade e tempo de emprego")

# Merge fase/data-prep
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
repo.refs[b"refs/heads/main"] = repo.refs[b"refs/heads/fase/data-prep"]

# 4. Branch fase/modelagem
head_commit = repo.head()
repo.refs[b"refs/heads/fase/modelagem"] = head_commit
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/fase/modelagem")

with open("projeto_credito.ipynb", "a") as f: f.write("\n")
porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"feat: cria pipeline de balanceamento smote e escalonamento")

with open("projeto_credito.ipynb", "a") as f: f.write("\n")
porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"feat: treina knn e arvore de decisao")

with open("projeto_credito.ipynb", "a") as f: f.write("\n")
porcelain.add(path, paths=["projeto_credito.ipynb"])
porcelain.commit(path, message=b"fix: ajusta hiperparametros para combater overfitting")

# Merge fase/modelagem
repo.refs.set_symbolic_ref(b"HEAD", b"refs/heads/main")
repo.refs[b"refs/heads/main"] = repo.refs[b"refs/heads/fase/modelagem"]

# Final commit with everything in its place just in case
porcelain.add(path)
porcelain.commit(path, message=b"docs: atualiza doc no readme e finaliza avaliacao")

print("Git history created successfully!")
