from typing import Dict, List
import os
import json

from typing import Dict, List
import os
import json

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pygraphviz as pgv
import pygraphviz as pgv


class NodeIn(BaseModel):
    id: str
    freq: int
    freq: int


class EdgeIn(BaseModel):
    source: str
    target: str


class Options(BaseModel):
    engine: str = "sfdp"
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

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "graph_layout_service"}

@app.post("/graph/layout")
def graph_layout(req: LayoutRequest) -> Dict[str, Dict]:
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    cache_key = f"{req.dataset_id}_{req.options.engine}.json"
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        width = req.options.width
        height = req.options.height
        pad = req.options.padding
        positions: Dict[str, Dict[str, float]] = {}
        for node_id, pos in cached["positions_norm"].items():
            x = pad + pos["x"] * (width - 2 * pad)
            y = pad + pos["y"] * (height - 2 * pad)
            positions[node_id] = {"x": x, "y": y}
        width = req.options.width
        height = req.options.height
        pad = req.options.padding
        positions: Dict[str, Dict[str, float]] = {}
        for node_id, pos in cached["positions_norm"].items():
            x = pad + pos["x"] * (width - 2 * pad)
            y = pad + pos["y"] * (height - 2 * pad)
            positions[node_id] = {"x": x, "y": y}
        return {
            "positions": positions,
            "positions": positions,
            "bbox": {"minX": 0, "minY": 0, "maxX": width, "maxY": height},
        }

    graph = pgv.AGraph(strict=False, directed=False)

    for n in req.nodes:
        graph.add_node(n.id)

    for e in req.edges:
        graph.add_edge(e.source, e.target)

    # Ensure START participates in layout by tethering it (invisibly) to zero-indegree cluster nodes
    all_ids = set([n.id for n in req.nodes])
    for e in req.edges:
        all_ids.add(e.source)
        all_ids.add(e.target)
    indegree = {nid: 0 for nid in all_ids}
    for e in req.edges:
        indegree[e.target] = indegree.get(e.target, 0) + 1
        indegree.setdefault(e.source, 0)
    has_start = 'START' in all_ids
    if has_start:
        for nid, deg in indegree.items():
            if nid == 'START':
                continue
            if nid.startswith('response-'):
                continue
            if deg == 0:
                graph.add_edge('START', nid, style='invis')

    graph.graph_attr.update(overlap="true")
    graph.layout(prog=req.options.engine)

    raw_positions: Dict[str, Dict[str, float]] = {}
    xs: List[float] = []
    ys: List[float] = []
    for n in graph.nodes():
        pos = n.attr["pos"]
        if pos:
            parts = pos.split(",")
            x = float(parts[0])
            y = float(parts[1])
            node_id = str(n)
            raw_positions[node_id] = {"x": x, "y": y}
            xs.append(x)
            ys.append(y)

    min_x = min(xs) if xs else 0.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0

    span_x = max_x - min_x if (max_x - min_x) != 0 else 1.0
    span_y = max_y - min_y if (max_y - min_y) != 0 else 1.0

    positions_norm: Dict[str, Dict[str, float]] = {}
    for node_id, pos in raw_positions.items():
        nx = (pos["x"] - min_x) / span_x
        ny = (pos["y"] - min_y) / span_y
        positions_norm[node_id] = {"x": nx, "y": ny}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"positions_norm": positions_norm}, f)

    width = req.options.width
    height = req.options.height
    pad = req.options.padding
    positions: Dict[str, Dict[str, float]] = {}
    for node_id, pos in positions_norm.items():
        x = pad + pos["x"] * (width - 2 * pad)
        y = pad + pos["y"] * (height - 2 * pad)
        positions[node_id] = {"x": x, "y": y}
    graph = pgv.AGraph(strict=False, directed=False)

    for n in req.nodes:
        graph.add_node(n.id)

    for e in req.edges:
        graph.add_edge(e.source, e.target)

    # Ensure START participates in layout by tethering it (invisibly) to zero-indegree cluster nodes
    all_ids = set([n.id for n in req.nodes])
    for e in req.edges:
        all_ids.add(e.source)
        all_ids.add(e.target)
    indegree = {nid: 0 for nid in all_ids}
    for e in req.edges:
        indegree[e.target] = indegree.get(e.target, 0) + 1
        indegree.setdefault(e.source, 0)
    has_start = 'START' in all_ids
    if has_start:
        for nid, deg in indegree.items():
            if nid == 'START':
                continue
            if nid.startswith('response-'):
                continue
            if deg == 0:
                graph.add_edge('START', nid, style='invis')

    graph.graph_attr.update(overlap="false")
    graph.layout(prog=req.options.engine)

    raw_positions: Dict[str, Dict[str, float]] = {}
    xs: List[float] = []
    ys: List[float] = []
    for n in graph.nodes():
        pos = n.attr["pos"]
        if pos:
            parts = pos.split(",")
            x = float(parts[0])
            y = float(parts[1])
            node_id = str(n)
            raw_positions[node_id] = {"x": x, "y": y}
            xs.append(x)
            ys.append(y)

    min_x = min(xs) if xs else 0.0
    max_x = max(xs) if xs else 1.0
    min_y = min(ys) if ys else 0.0
    max_y = max(ys) if ys else 1.0

    span_x = max_x - min_x if (max_x - min_x) != 0 else 1.0
    span_y = max_y - min_y if (max_y - min_y) != 0 else 1.0

    positions_norm: Dict[str, Dict[str, float]] = {}
    for node_id, pos in raw_positions.items():
        nx = (pos["x"] - min_x) / span_x
        ny = (pos["y"] - min_y) / span_y
        positions_norm[node_id] = {"x": nx, "y": ny}

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"positions_norm": positions_norm}, f)

    width = req.options.width
    height = req.options.height
    pad = req.options.padding
    positions: Dict[str, Dict[str, float]] = {}
    for node_id, pos in positions_norm.items():
        x = pad + pos["x"] * (width - 2 * pad)
        y = pad + pos["y"] * (height - 2 * pad)
        positions[node_id] = {"x": x, "y": y}

    return {
        "positions": positions_norm,
        "bbox": {"minX": 0, "minY": 0, "maxX": width, "maxY": height},
        "positions": positions_norm,
        "bbox": {"minX": 0, "minY": 0, "maxX": width, "maxY": height},
    }