from typing import Dict, List
import os
import json
import random

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HAS_PYGRAPHVIZ = False
try:
    import pygraphviz as pgv
    HAS_PYGRAPHVIZ = True
except ImportError:
    print("pygraphviz not installed. Will use random layout fallback (messier).")


class NodeIn(BaseModel):
    id: str
    freq: int


class EdgeIn(BaseModel):
    source: str
    target: str


class Options(BaseModel):
    engine: str = "sfdp"
    width: int = 1200
    height: int = 800
    padding: int = 20


class LayoutRequest(BaseModel):
    dataset_id: str
    nodes: List[NodeIn]
    edges: List[EdgeIn]
    options: Options = Options()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cache_path(dataset_id: str, engine: str) -> str:
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{dataset_id}_{engine}.json")


def _load_cache(path: str) -> dict | None:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_cache(path: str, positions_norm: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"positions_norm": positions_norm}, f)


def _normalize_positions(raw_positions: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    xs = [p["x"] for p in raw_positions.values()]
    ys = [p["y"] for p in raw_positions.values()]
    min_x, max_x = (min(xs) if xs else 0.0), (max(xs) if xs else 1.0)
    min_y, max_y = (min(ys) if ys else 0.0), (max(ys) if ys else 1.0)
    span_x = max_x - min_x if (max_x - min_x) != 0 else 1.0
    span_y = max_y - min_y if (max_y - min_y) != 0 else 1.0
    return {
        nid: {"x": (p["x"] - min_x) / span_x, "y": (p["y"] - min_y) / span_y}
        for nid, p in raw_positions.items()
    }


def _scale_positions(positions_norm: dict, width: int, height: int, pad: int) -> Dict[str, Dict[str, float]]:
    return {
        nid: {"x": pad + p["x"] * (width - 2 * pad), "y": pad + p["y"] * (height - 2 * pad)}
        for nid, p in positions_norm.items()
    }


def _generate_random_layout(req: LayoutRequest) -> Dict[str, Dict[str, float]]:
    print("WARNING: Using random layout fallback. Install pygraphviz for better graph layouts.")
    raw_positions = {}
    for n in req.nodes:
        raw_positions[n.id] = {"x": random.random(), "y": random.random()}
    return raw_positions


def _generate_pygraphviz_layout(req: LayoutRequest) -> Dict[str, Dict[str, float]]:
    graph = pgv.AGraph(strict=False, directed=False)

    for n in req.nodes:
        graph.add_node(n.id)
    for e in req.edges:
        graph.add_edge(e.source, e.target)

    # Tether START to zero-indegree cluster nodes
    all_ids = {n.id for n in req.nodes}
    for e in req.edges:
        all_ids.add(e.source)
        all_ids.add(e.target)
    indegree = {nid: 0 for nid in all_ids}
    for e in req.edges:
        indegree[e.target] = indegree.get(e.target, 0) + 1
        indegree.setdefault(e.source, 0)
    if "START" in all_ids:
        for nid, deg in indegree.items():
            if nid == "START" or nid.startswith("response-"):
                continue
            if deg == 0:
                graph.add_edge("START", nid, style="invis")

    graph.graph_attr.update(overlap="true")
    graph.layout(prog=req.options.engine)

    raw_positions: Dict[str, Dict[str, float]] = {}
    for n in graph.nodes():
        pos = n.attr["pos"]
        if pos:
            parts = pos.split(",")
            raw_positions[str(n)] = {"x": float(parts[0]), "y": float(parts[1])}

    return raw_positions


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "graph_layout_service"}


@app.post("/graph/layout")
def graph_layout(req: LayoutRequest) -> Dict[str, Dict]:
    cache_file = _cache_path(req.dataset_id, req.options.engine)
    cached = _load_cache(cache_file)

    if cached and "positions_norm" in cached:
        positions_norm = cached["positions_norm"]
    else:
        if HAS_PYGRAPHVIZ:
            raw_positions = _generate_pygraphviz_layout(req)
        else:
            raw_positions = _generate_random_layout(req)
        positions_norm = _normalize_positions(raw_positions)
        _save_cache(cache_file, positions_norm)

    width = req.options.width
    height = req.options.height
    pad = req.options.padding

    return {
        "positions": positions_norm,
        "bbox": {"minX": 0, "minY": 0, "maxX": width, "maxY": height},
    }
