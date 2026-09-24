## CI1 : Premiers pas

Le but de cette séance est de mettre en place l'environnement de travail et de pratiquer les notions vues en cours.
Nous allons voir comment utiliser SLURM, créer un environnement virtuel avec Mamba, entraîner un modèle simple et
visualiser l'apprentissage.


**Modalités de rendu :** L'évaluation se fera sur la base d'un rapport au format Markdown (rapport.md).
Vous devez initialiser un dépôt Git, y créer un répertoire TP1, et y placer votre rapport ainsi que vos scripts.
Mettez à jour votre dépôt au fur et à mesure de votre progression. Le lien de ce dépôt Git devra être envoyé à votre enseignant à la fin du TP.


### Utilisation de SLURM (∼30mn, – facile)

Nous allons accéder à des GPUs en utilisant un logiciel appelé **Slurm**.
Slurm est un ordonnanceur qui partage équitablement des ressources (CPUs, mémoire, GPUs) entre de nombreux utilisateurs.
L’idée est simple : vous _demandez_ des ressources; si elles sont disponibles, Slurm vous _attribue_ une allocation dans laquelle vos commandes s’exécutent.


**⚠️ Ressources partagées :** pour ce TP, ne demandez pas plus de --cpus-per-task=1 & --mem=8G (sauf consigne contraire). Demander "tous les CPUs" ou "toute la mémoire" bloque les autres utilisateurs.


**Connexion via SSH**

Pour accéder aux machines contenant les GPUs, nous allons utiliser SSH. Comme la machine se trouve dans le réseau de l'école, on ne peut y accéder que depuis l'école, ou à travers le VPN, ou en passant par le portail SSH public. Demandez à votre enseignant l'adresse IP.
Si vous utilisez Linux ou Mac, configurez le fichier ~/.ssh/config de la manière suivante pour un accès facilité :



Host LAB\_GATEWAY
User VOTRE\_IDENTIFIANT\_TSP
Hostname ssh2.imtbs-tsp.eu
IdentitiesOnly yes
IdentityFile CHEMIN\_VERS\_CLEF\_SSH\_PRIVEE
ServerAliveInterval 60
ServerAliveCountMax 2

Host tsp-client
User VOTRE\_IDENTIFIANT\_TSP
HostName ADRESSE\_IP
IdentitiesOnly yes
IdentityFile CHEMIN\_VERS\_CLEF\_SSH\_PRIVEE
ServerAliveInterval 60
ServerAliveCountMax 2
ProxyJump LAB\_GATEWAY


Vous pourrez alors vous connecter en faisant ssh tsp-client. Si vous êtes sous Windows, utilisez WSL2 ou PuTTY.


La machine sur laquelle vous vous connectez est une machine de connexion ( _login_) peu puissante : n'y exécutez pas de programmes coûteux en ressources !

**Génération de clé SSH**

Il est probable que votre clef SSH ne soit pas encore enregistrée sur la machine distante.
Pour générer une clef privée, faites :


ssh-keygen -t ed25519 -C "votre.email@domaine"

Copiez ensuite la clef publique (~/.ssh/id\_ed25519.pub) sur la machine distante et collez-la dans le fichier ~/.ssh/authorized\_keys (créez-le s'il n'existe pas). À faire sur LAB\_GATEWAY et sur tsp-client.


**Premiers pas en mode interactif avec srun**

Exécutez nvidia-smi sur la machine de connexion : cette commande devrait échouer (il n'y a pas de GPU).
Demandez des ressources en mode interactif :


srun --partition=gpu --gres=gpu:1 --time=01:00:00 --cpus-per-task=1 --mem=8G --pty bash

**✍️ À mettre dans votre rapport (rapport.md) :**
Une fois connecté au nœud de calcul, exécutez nvidia-smi. Quel est le modèle exact du GPU qui vous a été alloué ?


**Observer et arrêter ses jobs : squeue, scancel**

Ouvrez un _nouveau terminal_ (sans fermer le précédent), connectez-vous au cluster et lancez :


squeue -u $USER

Identifiez le _JobID_ de votre job interactif et terminez-le avec scancel MON\_JOB\_ID.

**✍️ À mettre dans votre rapport :**
Notez la commande exacte que vous avez tapée pour annuler votre job.


**Soumettre un script non interactif avec sbatch**

Pour des entraînements longs, on utilise sbatch.
Complétez le script suivant (code à trous) pour qu'il demande 1 GPU, 1 CPU, 8 Go de RAM sur la partition GPU pour une durée d'une heure.

#!/bin/bash#SBATCH --partition=\_\_\_\_\_\_\_\_#SBATCH -t 01:00:00#SBATCH --gres=\_\_\_\_\_\_\_\_#SBATCH --cpus-per-task=\_\_\_\_\_\_\_\_#SBATCH --mem=\_\_\_\_\_\_\_\_#SBATCH -J hello-slurm#SBATCH -o logs/%x-%j.out#SBATCH -e logs/%x-%j.errset-euo pipefail
mkdir -p logs

echo "Job $SLURM\_JOB\_ID on $SLURM\_NODELIST"
nvidia-smi \|\| echo "nvidia-smi indisponible"
echo "Bonjour depuis SLURM !"

**✍️ Action :** Sauvegardez ce script sous hello.sh, soumettez-le avec sbatch hello.sh, et **ajoutez ce fichier script dans votre dépôt Git (dossier TP1)**.


**✍️ À mettre dans votre rapport :**
Lisez le fichier de log généré dans le dossier logs/. Quel est le nom exact de ce fichier ?


**Analyser ses jobs : scontrol & sacct**

Consultez l’historique d'un de vos jobs terminés :


sacct -j MON\_JOB\_ID --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS

**✍️ À mettre dans votre rapport :**
Expliquez brièvement avec vos propres mots la différence entre ReqMem et MaxRSS.


**Transférer des fichiers (avec ProxyJump)**

Depuis votre machine → vers le cluster :

scp ./mnist.zip tsp-client:~/data/

Depuis le cluster → vers votre machine :

scp tsp-client:~/results/output.log ./output.log

Synchroniser un dossier :

rsync -avhP ./results/ tsp-client:~/backup-results/

**Cheat-sheet (récapitulatif)**

- **Observation** : squeue -u $USER, scancel JOBID, sacct -j JOBID
- **Interactif** : srun --partition=gpu --gres=gpu:1 --time=01:00:00 --cpus-per-task=1 --mem=8G --pty bash
- **Batch** : sbatch script.sh
- **Diagnostics rapides** : nvidia-smi, htop

### Création d'un environnement virtuel Python (∼20mn, – facile)

Dans cette section, vous apprendrez à créer un environnement virtuel avec **Miniforge/Mamba** et à installer **PyTorch** avec le support **CUDA**.
L’objectif est d’obtenir une installation _isolée_ (pas de conflit avec le système), _reproductible_, et prête à exploiter les GPUs du cluster.


**IMPORTANT :** Pour installer des packages, il faut toujours utiliser une réservation de nœud de calcul (idéalement avec un GPU). L'installation utilise beaucoup de ressources que la machine de connexion (login node) ne possède pas. De plus, certains packages (comme PyTorch) vérifient la présence du GPU lors de la configuration.


Lancez donc cette commande avant de continuer :

srun --partition=gpu --gres=gpu:1 --time=01:00:00 --cpus-per-task=1 --mem=8G --pty bash

**Installer Miniforge (conda-forge) et activer Mamba**

Exécutez les commandes suivantes pour télécharger et installer Miniforge :

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

Répondez _yes_ lorsqu’une question vous demande une confirmation. Pour que les modifications soient prises en compte, déconnectez-vous de votre session SSH puis reconnectez-vous (et relancez votre srun !).

Vous devriez maintenant pouvoir exécuter mamba. Si la commande n'est pas trouvée, exécutez temporairement : source ~/miniforge3/etc/profile.d/conda.sh.

**Créer et activer un environnement virtuel**

Créez un environnement dédié au cours (nommé deeplearning, avec Python 3.10) :

mamba create -n deeplearning python=3.10

Activez l’environnement nouvellement créé :

mamba activate deeplearning

Vous devrez _toujours_ activer cet environnement en arrivant sur le cluster avant de travailler sur les TPs.

**✍️ À mettre dans votre rapport :**
Quelle commande utilisez-vous pour vérifier la version exacte de Python installée dans votre environnement actif, ainsi que le chemin du binaire utilisé ? Notez la commande et son résultat.


**Installer PyTorch (GPU) + TensorBoard**

Installez PyTorch avec le support CUDA 12.1 en utilisant Mamba. Les canaux (-c) sont indispensables pour récupérer la bonne variante.


mamba install pytorch torchvision torchaudio pytorch-cuda=12.1-c pytorch -c nvidia
mamba install tensorboard -c conda-forge

**Vérifier l’installation PyTorch + CUDA**

Complétez le script Python ci-dessous (code à trous) pour vérifier que PyTorch détecte bien le GPU. Enregistrez-le sous check\_gpu.py.

import \_\_\_\_\_\_\_\_

print("PyTorch version:", torch.\_\_version\_\_)
gpu\_available = torch.cuda.\\_\\_\\_\_\_\_\_\_()print("CUDA available:", gpu\_available)if gpu\_available:print("Device count:", torch.cuda.device\_count())print("Device 0 name:", torch.cuda.get\_device\_name(0))else:print("Attention, aucun GPU détecté !")

**✍️ Action :** Exécutez ce script via python check\_gpu.py.


**✍️ À mettre dans votre rapport :**
Copiez-collez la sortie de ce script. Si CUDA available retourne False, listez dans votre rapport deux raisons possibles qui pourraient expliquer ce problème.


**Rendre l’environnement reproductible**

Pour garantir que votre code fonctionnera partout de la même manière, il est crucial de versionner les dépendances de votre projet.

mamba env export --from-history -n deeplearning > environment.yml

**✍️ Action :** **Ajoutez ce fichier environment.yml dans le dossier TP1 de votre dépôt Git** et commitez-le.


**Bonus — Vérifier TensorBoard**

TensorBoard sera utilisé plus tard pour visualiser vos courbes d'apprentissage.

**✍️ À mettre dans votre rapport :**
Quelle commande permet d'afficher la version de TensorBoard que vous venez d'installer ?


### Exercices théoriques (Papier & Markdown) (∼30mn, – moyen)

Dans cette section, vous allez consolider vos connaissances sur les concepts vus en cours
(perceptron multicouche, rétropropagation, descente de gradient, fonctions d'activation).


**Modalités de rendu :** Pour ces questions, vous pouvez rédiger vos réponses mathématiques directement dans votre rapport.md en utilisant la syntaxe LaTeX (ex: $$ Y = X W^T + b $$). Pour les schémas, n'hésitez pas à les dessiner sur papier, les prendre en photo, les placer dans votre dossier TP1 et les inclure dans votre Markdown (ex: !\[Graphe\](mon\_schema.jpg)).


**Conventions utilisées** : nous manipulons des _vecteurs-lignes_.
Ainsi, pour un _seul exemple_ d’entrée, X est de taille 1×3.
Pour un _batch_ de taille N, X est de taille N×3.


### Architecture et Paramètres

Dessinez (sur papier ou outil numérique) un perceptron multicouche (MLP) avec :

- Une couche d’entrée avec 3 neurones
- Une couche cachée avec 4 neurones
- Une couche de sortie avec 2 neurones

**✍️ À mettre dans votre rapport :**
Insérez l'image de votre schéma. Indiquez ensuite le nombre total de paramètres de ce modèle, d'abord sans prendre en compte les biais, puis avec les biais. Détaillez votre calcul (ex: Couche 1 = X \* Y = Z ...).


**Équations et dimensions (Texte à trous)**

Recopiez et complétez les équations du _forward pass_ dans votre rapport pour un batch d’exemples X de taille N×3. Remplacez les ? par les bonnes dimensions.

```
H = ReLU( X · W1^T + b1 )
Y = H · W2^T + b2

Dimensions :
X  : (N, 3)
W1 : (?, ?)
b1 : (1, ?) -> diffusé en (N, ?)
H  : (?, ?)
W2 : (?, ?)
b2 : (1, ?) -> diffusé en (N, ?)
Y  : (?, ?)
```

### Graphe de calcul et Rétropropagation

Considérez la fonction f(x, y, z) = x / y + z.

**✍️ À mettre dans votre rapport :**

1. Dessinez le graphe de calcul de cette fonction (identifiez un nœud intermédiaire q = x / y).
2. Effectuez le _forward pass_ avec x = 2, y = 4, z = 0. Quelle est la valeur de f ?
3. Effectuez la _backpropagation_ pour calculer les gradients locaux : ∂f/∂x, ∂f/∂y, et ∂f/∂z au point donné. Détaillez les étapes de votre calcul.

**Mise à jour des poids**

En reprenant les résultats de la question précédente, effectuez une étape de descente de gradient avec un _learning rate_ η = 1.

**✍️ À mettre dans votre rapport :**
Calculez les nouvelles valeurs de x', y', et z'.
Calculez la nouvelle sortie f' = f(x', y', z'). La valeur de la fonction a-t-elle diminué comme attendu ?


### Questions de réflexion

**✍️ À mettre dans votre rapport :** Répondez brièvement (2-3 phrases maximum par question) :


- Pourquoi utilisons-nous la règle de la chaîne (chain rule) pour calculer les gradients dans les réseaux de neurones profonds ?
- Quelles sont les principales raisons d'utiliser des _mini-batchs_ plutôt que d'optimiser sur un seul exemple à la fois ou sur l’ensemble total des données ?

**Association (Texte à trous)**

Associez la bonne couche de sortie et la bonne fonction de perte usuelle pour chaque tâche dans votre rapport :

```
Tâche                   | Fonction finale (Sortie) | Fonction de perte (Loss)
------------------------|--------------------------|---------------------------
Classification binaire  | 1. ___________           | A. ___________
Classification multi    | 2. ___________           | B. ___________
Régression pure         | 3. Identité (aucune)     | C. MSE (Mean Squared Error)
```

### Votre premier réseau de neurones (∼45mn, – moyen)

Dans cette section, vous allez implémenter un réseau de neurones simple avec PyTorch, l’entraîner sur le jeu de données d'images CIFAR-10, puis l’évaluer proprement.


### Étape 1 : Préparation des données

Complétez le code ci-dessous pour télécharger et charger le dataset CIFAR-10 avec une normalisation _canonique_. Placez ce code dans un fichier train.py.

```
import os
import torch
import torchvision
from torchvision import transforms, datasets

# Normalisation "classique" pour CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([\
    transforms.ToTensor(),\
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),\
])

# CHARGEMENT DES DONNÉES (à compléter)
trainset = datasets.CIFAR10(root='./data', train=_____, download=True, transform=transform)
testset  = datasets.CIFAR10(root='./data', train=_____, download=True, transform=transform)

# Ne pas monopoliser les CPUs sur Slurm
def get_num_workers(default=2, cap=4):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default
    return max(0, min(cap, n))

num_workers = get_num_workers()

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=_____, num_workers=num_workers, pin_memory=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=_____, num_workers=num_workers, pin_memory=True
)
```

Les num\_workers accélèrent le préchargement en utilisant du CPU, mais **n’en mettez pas trop** (vous partagez la machine).
pin\_memory=True accélère la copie CPU→GPU.


**✍️ À mettre dans votre rapport :**
Expliquez brièvement à quoi servent les arguments batch\_size et shuffle dans le DataLoader. Pourquoi shuffle doit-il avoir une valeur différente pour l'entraînement et pour le test ?


### Étape 2 : Implémentation du réseau

Dans votre fichier train.py, complétez l'implémentation d'un Perceptron Multicouche (MLP) avec une couche cachée de 128 neurones pour des images 32×32×3 (RGB) et 10 classes.

```
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Image 32x32 avec 3 canaux (RGB) -> aplatie
        self.fc1 = nn.Linear(________, 128)  # couche cachée
        self.fc2 = nn.Linear(128, ____)      # couche de sortie (logits)

    def forward(self, x):
        # Aplatir de manière robuste en préservant la dimension batch (dim 0) :
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                     # NE PAS appliquer Softmax ici
        return x
```

**✍️ À mettre dans votre rapport :**

1. Dans la méthode forward, pourquoi utilise-t-on torch.flatten(x, 1) avant de passer les données à la couche linéaire ?
2. Pourquoi est-il crucial de **ne pas** ajouter de fonction d'activation Softmax à la fin de notre réseau quand on s'apprête à utiliser nn.CrossEntropyLoss dans PyTorch ?

### Étape 3 : Entraînement du modèle

Configurez la boucle d'entraînement. Complétez les étapes de la passe arrière ( _backward pass_) et de l'optimiseur ( _code à trous_).

```
import random
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train() # Passe le modèle en mode entraînement
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for inputs, labels in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 1. Réinitialiser les gradients
        optimizer.________(set_to_none=True)

        # 2. Passe avant (forward)
        outputs = model(______)

        # 3. Calcul de la perte
        loss = criterion(outputs, ______)

        # 4. Passe arrière (Rétropropagation)
        loss.________()

        # 5. Mise à jour des poids
        optimizer.________()

        # Statistiques
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total   += labels.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc  = running_correct / running_total
    print(f"Epoch {epoch+1:02d} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")
```

**✍️ Action :** Exécutez ce script via Slurm (à l'aide de sbatch ou srun).


**✍️ À mettre dans votre rapport :**
Quelle est la différence fondamentale entre optimizer.zero\_grad() et loss.backward() ?


### Étape 4 : Évaluation sur l’ensemble de test

Ajoutez le code d'évaluation à la fin de votre script train.py.

```
model.eval() # Mode évaluation
classes = trainset.classes

total = 0
correct = 0

with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

acc = correct / total
print(f"Test accuracy: {acc:.3f}")
```

**✍️ À mettre dans votre rapport :**

1. Pourquoi utilise-t-on le bloc with torch.no\_grad(): lors de l'évaluation ? Quel est l'avantage en termes de ressources matérielles ?
2. Si votre classificateur prédisait les classes de manière purement aléatoire, à quelle précision (accuracy) environ devriez-vous vous attendre sur le jeu de test CIFAR-10 ?

### Étape 5 : Sauvegarde et chargement du modèle

Après l'évaluation, il est recommandé de sauvegarder les poids de votre modèle pour pouvoir le réutiliser sans le ré-entraîner.

```
# Sauvegarde des poids
torch.save(model.state_dict(), "mlp_model.pth")

# Exemple de chargement (sur CPU) - Vous pouvez tester cela dans un script séparé
# model2 = MLP().to("cpu")
# state = torch.load("mlp_model.pth", map_location="cpu", weights_only=True)
# model2.load_state_dict(state)
# model2.eval()
```

**✍️ Action finale :** **Commitez et pushez votre fichier train.py dans le répertoire TP1 de votre dépôt Git.**

### Utilisation de TensorBoard (∼45mn, – moyen)

Dans cet exercice, vous allez instrumenter votre entraînement pour **journaliser** proprement les métriques
et explorer l’interface de **TensorBoard** afin de diagnostiquer le bruit, le sur-apprentissage, et comparer plusieurs jeux d’hyperparamètres.
Nous distinguerons clairement **train**, **validation** et **test** : on suit et on ajuste avec _train/val_,
et on ne consulte _test_ qu’à la toute fin.


**Pourquoi val → test ?** Pendant le développement, on évalue et on choisit les hyperparamètres avec un _ensemble de validation_.
Le test est réservé à la mesure finale, pour éviter de “sur-apprendre” (overfitter) au protocole d’évaluation.


### Étape 1 — Préparation : Split et hyperparamètres

Créez un nouveau script train\_tb.py dans votre dossier TP1. Commencez par définir un dossier de logs dynamique et séparer l'ensemble d'entraînement en train/val (code à trous).

```
import os, random, datetime, torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Hyperparamètres (faciles à modifier pour nos futures expériences)
hparams = dict(model="MLP", batch_size=32, lr=1e-2, seed=0, weight_decay=0.0)

# 1. Création d'un nom de dossier unique (modèle + hparams + timestamp)
run_name = f"{hparams['model']}/bs{hparams['batch_size']}_lr{hparams['lr']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
logdir = os.path.join("runs", _______)
print("Logdir:", logdir)

# Instanciation du SummaryWriter
writer = SummaryWriter(log_dir=_______)

# 2. Séparation du jeu de données (on suppose 'trainset' déjà chargé comme à l'exercice précédent)
N = len(trainset)
val_size = int(0.1 * N)
train_size = N - _______

# Utilisation de random_split avec une graine fixe pour la reproductibilité
train_subset, val_subset = random_split(trainset, [_______, _______], generator=torch.Generator().manual_seed(0))

# Création des DataLoaders (à compléter avec hparams)
trainloader = DataLoader(train_subset, batch_size=hparams["_______"], shuffle=True, pin_memory=True)
valloader   = DataLoader(val_subset,   batch_size=hparams["_______"], shuffle=False, pin_memory=True)
```

**✍️ À mettre dans votre rapport :**
Pourquoi est-il important d'inclure la date, l'heure et les hyperparamètres dans le nom du dossier de logs (run\_name) ?


### Étape 2 — Calcul des métriques par époque

Il ne faut pas évaluer la perte d'une époque uniquement sur le dernier batch. Ajoutez cette fonction pour calculer une vraie moyenne sur tout un DataLoader :

```
@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

    loss_avg = loss_sum / total
    acc = correct / total
    return loss_avg, acc
```

### Étape 3 — Instrumenter l’entraînement

Intégrez writer.add\_scalar dans votre boucle d'entraînement pour logger la perte au niveau du batch, et les métriques au niveau de l'époque.

```
# Initialisation (modèle, optimizer, criterion...) identiques à l'exercice précédent.
global_step = 0
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss_sum, running_total = 0.0, 0

    for b, (x, y) in enumerate(trainloader):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Logging batch (toutes les 10 itérations pour ne pas surcharger)
        if b % 10 == 0:
            writer.add_scalar("Loss/train_step", _______, global_step)

        optimizer.step()
        running_loss_sum += loss.item() * y.size(0)
        running_total    += y.size(0)
        global_step += 1

    # Métriques de fin d'époque
    train_loss = running_loss_sum / running_total
    val_loss, val_acc = epoch_metrics(valloader, model, criterion, device)

    # Logging époque
    writer.add_scalar("Loss/train", _______, epoch)
    writer.add_scalar("Loss/val",   _______, epoch)
    writer.add_scalar("Accuracy/val", _______, epoch)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

# Fin d'entraînement
writer._______() # Force l'écriture des données sur le disque
```

### Étape 4 — Visualiser TensorBoard

Une fois votre script exécuté, un dossier runs/ a été créé. Pour visualiser l'interface, vous devez rapatrier ces données sur votre machine locale ou créer un tunnel SSH.

**Méthode recommandée (transfert local) :**

```
# Depuis un terminal sur VOTRE machine (pas sur le cluster) :
scp -r tsp-client:~/Chemin/Vers/Votre/TP1/runs ./runs
# Lancez ensuite TensorBoard localement :
tensorboard --logdir=runs
```

Ouvrez l’URL indiquée (généralement http://localhost:6006).

**✍️ À mettre dans votre rapport :**
Dans l’onglet _Scalars_, réglez le curseur **Smoothing** pour lisser la courbe Loss/train\_step.
À quel niveau de smoothing distinguez-vous clairement la _tendance_ sans masquer des changements importants ? Pourquoi observe-t-on autant de bruit sur Loss/train\_step comparativement à Loss/train ?


### Étape 5 — Mini-sweep d’hyperparamètres & diagnostic d’overfit

Modifiez le dictionnaire hparams dans votre script et lancez au moins **3 entraînements (runs) différents** :

- Run 1 : LR = 1e-2, batch\_size = 32
- Run 2 : LR = 1e-3, batch\_size = 32
- Run 3 : LR = 1e-1, batch\_size = 128

**✍️ À mettre dans votre rapport :**

1. Analysez les courbes Loss/train et Loss/val superposées pour ces 3 runs. Lequel donne la meilleure accuracy en validation ?
2. Comment détecte-t-on visuellement un **sur-apprentissage (overfitting)** sur les courbes de perte (loss) d'entraînement et de validation ? Décrivez l'allure des courbes.

**✍️ Action finale :** **Commitez et pushez votre script train\_tb.py, votre dossier runs/ (si léger, sinon ignorez-le via .gitignore) et votre rapport.md complété dans le répertoire TP1 de votre dépôt Git. N'oubliez pas d'envoyer le lien à votre enseignant.**

Ce TP vous a guidés de bout en bout : accéder à des GPUs partagés avec Slurm, travailler dans un environnement Python isolé et reproductible, rappeler les fondamentaux (architecture MLP, dimensions, passes avant/arrière, mini-batch), puis implémenter un premier entraînement complet et l’instrumenter avec TensorBoard.


Retenez surtout les bonnes pratiques : sur un cluster, on reste sobre dans les ressources demandées et on surveille ses jobs ; côté code, on structure une boucle d’apprentissage simple et robuste (train/val/test bien séparés), et on lit l’apprentissage dans TensorBoard plutôt que « au feeling ».


**N'oubliez pas de commiter vos dernières modifications (rapport et code) dans votre dossier TP1, de faire un push sur votre dépôt, et d'envoyer le lien à votre enseignant.** Gardez ce TP comme référence pour les prochaines séances.