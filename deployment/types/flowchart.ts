export interface Node {
    cluster_id: string;
    freq: number;
    representative_sentence: string;
    mean_similarity: number;
    sentences: Sentence[];
}

export interface Sentence {
    text: string;
    count: number;
    rollout_ids: number[];
}

export interface Edge {
    node_a: string;
    node_b: string;
}

export interface Rollout {
    index: string;
    edges: Edge[];
    correctness?: boolean;
}

export interface RolloutObject {
    edges: Edge[];
    correctness?: boolean;
}

export interface FlowchartData {
    nodes: Node[];
    responses?: { [key: string]: RolloutObject } | Rollout[];
}