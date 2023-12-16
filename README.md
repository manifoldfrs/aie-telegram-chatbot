```bash
python -m venv venv

source venv/bin/activate

mv .env.example .env

pip install -r requirements.txt
```

### Some issues I ran into
```Could not import libcairo-2```
This happened to me b/c I'm on an M1 Mac, just run ```brew install cairo```
