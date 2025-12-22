import os
import json
import sys
from typing import List, Callable, Dict, Any


class GeneCacheManager:
    def __init__(
        self, 
        cache_dir: str = "gene_cache", 
        registry_file: str = "registry.json", 
        base_path: str = None
    ):
        if base_path is None:
            try:
                base_path = __file__
            except NameError:
                base_path = os.getcwd()
        
        if os.path.isfile(base_path):
            base_dir = os.path.dirname(os.path.abspath(base_path))
        else:
            base_dir = os.path.abspath(base_path)

        if not os.path.isabs(cache_dir):
            self.cache_dir = os.path.join(base_dir, cache_dir)
        else:
            self.cache_dir = cache_dir

        self.registry_path = os.path.join(self.cache_dir, registry_file)
        self.recipes: Dict[str, Any] = {}
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}


    def load(self, key: str, func: Callable, update: bool = False, **kwargs) -> List[str]:
        self.register_recipe(key=key, func=func, **kwargs)
        return self.get(key=key, update=update)


    def register_recipe(self, key: str, func: Callable, **kwargs):
        self.recipes[key] = {'func': func, 'kwargs': kwargs}


    def get(self, key: str, update: bool = False) -> List[str]:
        if not update and key in self.registry:
            filename = self.registry[key]
            file_path = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(file_path):
                print(f"[Cache Hit] Loading '{key}'...")
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f.readlines()]
            else:
                print(f"[Cache Broken] File missing for '{key}'. Regenerating...")

        if key in self.recipes:
            print(f"[Generating] Running recipe for '{key}'... (Update={update})")
            recipe = self.recipes[key]
            
            try:
                genes = recipe['func'](**recipe['kwargs'])
            except Exception as e:
                print(f"[Error] Failed to generate '{key}': {e}")
                return []
            
            filename = f"{key}.txt"
            save_path = os.path.join(self.cache_dir, filename)
            
            with open(save_path, 'w') as f:
                f.write("\n".join(genes))
            
            self.registry[key] = filename 
            self._save_registry()
            
            print(f" -> Saved to {save_path}")
            return genes

        print(f"[NotFound] No cache and no recipe for '{key}'.")
        return []


    def _save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)


    def clear_cache(self, key: str):
        if key in self.registry:
            filename = self.registry[key]
            path = os.path.join(self.cache_dir, filename)
            if os.path.exists(path):
                os.remove(path)
            del self.registry[key]
            self._save_registry()
            print(f"Cache cleared for '{key}'")
