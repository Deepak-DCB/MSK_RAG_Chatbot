from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_vocab import SCHEMA_VERSION


SUPPORT_RANK = {
    "direct": 5,
    "indirect": 4,
    "inferred_from_same_section": 3,
    "inferred_from_path": 2,
    "weak": 1,
    "unsupported": 0,
}


PATH_FAMILIES = [
    {
        "name": "TOS scapular depression pathway",
        "steps": [
            ["scapular depression"],
            ["clavicular depression", "clavicle"],
            ["costoclavicular narrowing", "costoclavicular space"],
            ["brachial plexus compression", "brachial plexus", "subclavian artery", "subclavian vein", "neurovascular compression"],
            ["paresthesia", "neuralgia", "numbness", "tingling", "ulnar neuralgia", "TOS-like symptoms"],
        ],
    },
    {
        "name": "Scalene pathway",
        "steps": [
            ["scalene dysfunction", "scalene"],
            ["first rib mechanics", "first rib"],
            ["interscalene triangle", "costoclavicular space", "thoracic outlet"],
            ["brachial plexus compression", "brachial plexus"],
            ["TOS-like symptoms", "paresthesia", "numbness", "tingling"],
        ],
    },
    {
        "name": "Breathing and posture pathway",
        "steps": [
            ["swayback posture", "forward head posture", "slouching"],
            ["altered breathing mechanics", "belly breathing", "thoracic breathing"],
            ["scalene dysfunction", "scalene"],
            ["thoracic outlet mechanics", "thoracic outlet", "costoclavicular space", "interscalene triangle"],
            ["TOS-like symptoms", "paresthesia", "numbness", "tingling"],
        ],
    },
    {
        "name": "Cervical plexus / levator pathway",
        "steps": [
            ["levator scapulae", "middle scalene", "scalene"],
            ["cervical plexus entrapment", "cervical plexus"],
            ["neck pain", "headache", "migraine", "dizziness", "globus"],
        ],
    },
    {
        "name": "Secondary distal entrapment pathway",
        "steps": [
            ["thoracic outlet syndrome", "brachial plexus irritation", "brachial plexus"],
            ["sensitized nervous chain"],
            ["distal nerve vulnerability"],
            ["radial nerve", "ulnar nerve", "median nerve", "axillary nerve", "ulnar neuralgia", "paresthesia"],
        ],
    },
    {
        "name": "Dorsal scapular nerve pain pathway",
        "steps": [
            ["thoracic outlet syndrome", "costoclavicular space", "thoracic outlet"],
            ["dorsal scapular nerve"],
            ["pain between shoulder blades", "neck pain", "neuralgia"],
        ],
    },
    {
        "name": "Accessory nerve upper cervical pathway",
        "steps": [
            ["jugular foramen", "atlas", "jugular outlet obstruction"],
            ["accessory nerve"],
            ["sternocleidomastoid", "trapezius"],
        ],
    },
    {
        "name": "Occipital nerve neuralgia pathway",
        "steps": [
            ["suboccipitals", "trapezius", "cervical plexus"],
            ["occipital nerves"],
            ["neuralgia", "headache", "neck pain"],
            ["nerve block", "Tinel's sign", "strengthening"],
        ],
    },
    {
        "name": "Atlas instability screening pathway",
        "steps": [
            ["ligament laxity", "significant trauma"],
            ["atlas instability", "atlantoaxial instability", "craniocervical instability"],
            ["vertebral artery", "internal jugular vein", "brainstem compression", "jugular outlet obstruction"],
            ["dizziness", "headache", "dyspnea", "progressive neurologic deficit", "brainstem compromise"],
        ],
    },
    {
        "name": "AAI CCI brainstem red flag pathway",
        "steps": [
            ["atlantoaxial instability", "craniocervical instability"],
            ["brainstem compression"],
            ["brainstem compromise", "progressive neurologic deficit"],
        ],
    },
    {
        "name": "Vertebrobasilar red flag pathway",
        "steps": [
            ["vertebrobasilar insufficiency"],
            ["dizziness"],
            ["progressive neurologic deficit", "brainstem compromise"],
        ],
    },
    {
        "name": "Intracranial pressure venous pathway",
        "steps": [
            ["venous outflow impairment", "venous sinus stenosis", "jugular outlet obstruction", "internal jugular vein"],
            ["intracranial hypertension", "CSF pressure"],
            ["headache", "migraine", "tinnitus", "vestibular symptoms", "papilledema"],
        ],
    },
    {
        "name": "Intracranial hypertension papilledema screening pathway",
        "steps": [
            ["venous sinus stenosis", "internal jugular vein", "venous outflow impairment"],
            ["intracranial hypertension"],
            ["papilledema"],
        ],
    },
    {
        "name": "CSF pressure vestibular pathway",
        "steps": [
            ["CSF leak"],
            ["CSF pressure"],
            ["headache", "dizziness", "vestibular symptoms"],
        ],
    },
    {
        "name": "POTS craniovascular pathway",
        "steps": [
            ["thoracic outlet syndrome", "vascular TOS", "subclavian artery", "internal jugular vein"],
            ["craniovascular hyperperfusion", "venous outflow impairment"],
            ["POTS", "orthostatic symptoms", "fatigue", "brain fog"],
        ],
    },
    {
        "name": "Vascular TOS autonomic pathway",
        "steps": [
            ["vascular TOS", "subclavian artery", "subclavian vein"],
            ["sympathetic involvement", "craniovascular hyperperfusion"],
            ["autonomic symptoms"],
        ],
    },
    {
        "name": "POTS symptoms pathway",
        "steps": [
            ["craniovascular hyperperfusion", "venous outflow impairment"],
            ["POTS"],
            ["orthostatic symptoms", "autonomic symptoms", "fatigue", "brain fog"],
        ],
    },
    {
        "name": "POTS orthostatic symptoms pathway",
        "steps": [
            ["craniovascular hyperperfusion"],
            ["POTS"],
            ["orthostatic symptoms"],
        ],
    },
    {
        "name": "TMJ auditory pathway",
        "steps": [
            ["forward head posture", "mandibular retraction", "temporomandibular dysfunction"],
            ["temporomandibular joint", "mandible", "pterygoids"],
            ["trigeminal nerve", "facial nerve", "glossopharyngeal nerve", "eustachian tube", "tympanic plexus", "middle ear"],
            ["tinnitus", "ear pressure", "hearing symptoms", "jaw pain"],
        ],
    },
    {
        "name": "Trigeminal TMJ branch pathway",
        "steps": [
            ["mandibular retraction", "temporomandibular joint"],
            ["trigeminal nerve", "auriculotemporal nerve", "mandibular nerve"],
            ["lateral pterygoid", "medial pterygoid", "tensor tympani", "tensor veli palatini"],
            ["tinnitus", "jaw pain", "ear pressure", "hearing symptoms"],
            ["strengthening", "postural rehabilitation"],
        ],
    },
    {
        "name": "Phrenic cervical plexus breathing pathway",
        "steps": [
            ["cervical plexus", "scalene", "anterior scalene", "middle scalene"],
            ["phrenic nerve"],
            ["dyspnea", "altered breathing mechanics"],
            ["provocative testing"],
        ],
    },
    {
        "name": "ME CFS venous fatigue pathway",
        "steps": [
            ["venous outflow impairment", "intracranial hypertension"],
            ["ME/CFS"],
            ["fatigue", "brain fog"],
        ],
    },
    {
        "name": "Migraine venous pathway",
        "steps": [
            ["internal jugular vein", "thoracic outlet", "atlas"],
            ["venous outflow impairment"],
            ["migraine"],
        ],
    },
    {
        "name": "TMJ eustachian tinnitus pathway",
        "steps": [
            ["temporomandibular joint"],
            ["eustachian tube"],
            ["tinnitus", "ear pressure"],
        ],
    },
    {
        "name": "TMD jaw pain pathway",
        "steps": [
            ["mouth breathing", "tongue posture", "occlusion", "maxilla"],
            ["mandibular retraction"],
            ["temporomandibular joint"],
            ["disc displacement"],
            ["jaw pain"],
        ],
    },
    {
        "name": "Vestibular neck TMJ pathway",
        "steps": [
            ["suboccipitals", "atlas", "temporomandibular joint"],
            ["vertebral artery", "vestibulocochlear nerve", "middle ear", "vestibular system", "semicircular canals"],
            ["vestibular symptoms", "dizziness", "tinnitus"],
        ],
    },
    {
        "name": "Vestibular system symptoms pathway",
        "steps": [
            ["vestibulocochlear nerve", "semicircular canals", "temporomandibular joint"],
            ["vestibular system"],
            ["vestibular symptoms"],
        ],
    },
    {
        "name": "Lumbar plexus pelvic pathway",
        "steps": [
            ["psoas", "lumbar spine"],
            ["lumbar plexus", "lumbosacral plexus"],
            ["femoral nerve", "genitofemoral nerve", "ilioinguinal nerve", "lateral femoral cutaneous nerve"],
            ["groin pain", "pelvic pain", "hip pain", "neuralgia"],
        ],
    },
    {
        "name": "Pudendal nerve pelvic pain pathway",
        "steps": [
            ["piriformis", "obturator internus", "pelvic floor", "Alcock's canal"],
            ["pudendal nerve"],
            ["pudendal neuralgia"],
            ["pelvic pain"],
        ],
    },
    {
        "name": "Hypogastric plexus pelvic pain pathway",
        "steps": [
            ["pelvic floor", "psoas", "sacral plexus"],
            ["inferior hypogastric plexus"],
            ["inferior hypogastric plexopathy"],
            ["pelvic pain", "neuralgia"],
        ],
    },
    {
        "name": "Hip impingement glide pathway",
        "steps": [
            ["psoas", "posterior pelvic tilt", "swayback posture"],
            ["anterior femoral glide"],
            ["femoroacetabular impingement"],
            ["acetabulum", "labral impingement", "hip joint"],
            ["hip pain", "groin pain"],
        ],
    },
    {
        "name": "Knee posterior glide pathway",
        "steps": [
            ["posterior pelvic tilt", "swayback posture"],
            ["functional varus", "functional valgus", "tibial internal rotation", "tibial external rotation"],
            ["posterior tibial glide", "anterior tibial glide", "patella", "tibia"],
            ["knee pain", "patellofemoral pain syndrome", "jumper's knee"],
        ],
    },
    {
        "name": "Knee valgus patellofemoral pathway",
        "steps": [
            ["functional valgus"],
            ["tibial external rotation"],
            ["patella", "patellofemoral pain syndrome"],
            ["knee pain"],
        ],
    },
    {
        "name": "Knee anterior glide jumper pathway",
        "steps": [
            ["anterior tibial glide"],
            ["jumper's knee"],
            ["knee pain"],
        ],
    },
    {
        "name": "Lumbar disc loading pathway",
        "steps": [
            ["posterior pelvic tilt", "swayback posture"],
            ["lumbar lordosis", "lumbar spine"],
            ["disc loading"],
            ["disc herniation"],
            ["nerve root"],
            ["low back pain"],
        ],
    },
    {
        "name": "Scapular shoulder impingement pathway",
        "steps": [
            ["scapular dyskinesis", "scapular depression", "anterior scapular tilt", "scapular winging"],
            ["subacromial space", "coracoid process", "acromion", "glenohumeral joint"],
            ["rotator cuff", "supraspinatus", "subscapularis", "shoulder impingement"],
            ["shoulder pain"],
        ],
    },
    {
        "name": "Clenching bracing pathway",
        "steps": [
            ["protective bracing", "manual muscle testing"],
            ["altered breathing mechanics", "breath holding", "dyspnea"],
            ["fatigue", "neuralgia", "neck pain"],
        ],
    },
    {
        "name": "Scapular motor control pathway",
        "steps": [
            ["scapular dyskinesis", "motor control"],
            ["strengthening"],
            ["shoulder pain"],
        ],
    },
]


def _edge_key(edge: Dict[str, Any]) -> Tuple[str, str]:
    return str(edge.get("source_node_id")), str(edge.get("target_node_id"))


def _find_edge_between(
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    edge_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source_id in source_ids:
        for target_id in target_ids:
            candidates.extend(edge_lookup.get((source_id, target_id), []))
            candidates.extend(edge_lookup.get((target_id, source_id), []))
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda e: (
            1 if str(e.get("clinical_risk")) == "high" else 0,
            1 if str(e.get("relation_type")) == "red_flag_for" else 0,
            SUPPORT_RANK.get(str(e.get("support_level")), 0),
            1 if str(e.get("relation_type")) != "mentioned_with" else 0,
            len(e.get("source_span_ids") or []),
        ),
        reverse=True,
    )[0]


def _policy_for_edges(edges: List[Dict[str, Any]]) -> str:
    levels = [str(edge.get("support_level") or "weak") for edge in edges]
    risks = [str(edge.get("clinical_risk") or "low") for edge in edges]
    relations = {str(edge.get("relation_type")) for edge in edges}
    if "high" in risks:
        return "requires_red_flag_screening"
    if any(level in {"weak", "inferred_from_same_section", "inferred_from_path", "unsupported"} for level in levels):
        return "do_not_present_as_causal_conclusion"
    if any(rel.startswith("may_") or rel in {"associated_with", "mentioned_with"} for rel in relations):
        return "explain_as_possible_indirect_mechanism"
    return "safe_to_explain_as_mechanism"


def _path_support(edges: List[Dict[str, Any]]) -> Tuple[str, str, str]:
    weakest = min(edges, key=lambda e: SUPPORT_RANK.get(str(e.get("support_level")), 0))
    return (
        str(weakest.get("edge_id")),
        str(weakest.get("support_level") or "weak"),
        "direct" if all(str(e.get("support_level")) == "direct" for e in edges) else "indirect",
    )


def build_mechanism_paths(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    name_to_id = {str(node.get("canonical_name")): str(node.get("node_id")) for node in nodes}
    id_to_name = {str(node.get("node_id")): str(node.get("canonical_name")) for node in nodes}
    edge_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for edge in edges:
        edge_lookup.setdefault(_edge_key(edge), []).append(edge)

    paths: List[Dict[str, Any]] = []
    seen = set()
    for family in PATH_FAMILIES:
        step_ids = [[name_to_id[name] for name in step if name in name_to_id] for step in family["steps"]]
        step_ids = [ids for ids in step_ids if ids]
        if len(step_ids) < 2:
            continue

        chosen_edges: List[Dict[str, Any]] = []
        chosen_nodes: List[str] = []
        for i in range(len(step_ids) - 1):
            edge = _find_edge_between(step_ids[i], step_ids[i + 1], edge_lookup)
            if edge is None:
                if len(chosen_edges) >= 2:
                    break
                chosen_edges = []
                chosen_nodes = []
                continue
            if not chosen_nodes:
                chosen_nodes.append(str(edge.get("source_node_id")))
            chosen_edges.append(edge)
            target_id = str(edge.get("target_node_id"))
            source_id = str(edge.get("source_node_id"))
            if chosen_nodes[-1] == target_id:
                chosen_nodes.append(source_id)
            else:
                chosen_nodes.append(target_id)

        if len(chosen_edges) < 2:
            continue
        edge_ids = [str(edge.get("edge_id")) for edge in chosen_edges]
        key = tuple(edge_ids)
        if key in seen:
            continue
        seen.add(key)

        weakest_edge_id, weakest_support_level, path_support = _path_support(chosen_edges)
        span_ids = sorted({sid for edge in chosen_edges for sid in (edge.get("source_span_ids") or [])})
        article_ids = sorted({aid for edge in chosen_edges for aid in (edge.get("source_article_ids") or [])})
        names = [id_to_name.get(node_id, node_id) for node_id in chosen_nodes]
        path_id = f"path_{len(paths) + 1:04d}"
        paths.append(
            {
                "path_id": path_id,
                "path_family": family.get("name"),
                "node_ids": chosen_nodes,
                "edge_ids": edge_ids,
                "path_text": " -> ".join(names),
                "path_support": path_support,
                "weakest_edge_id": weakest_edge_id,
                "weakest_support_level": weakest_support_level,
                "clinical_policy": _policy_for_edges(chosen_edges),
                "source_span_ids": span_ids,
                "source_article_ids": article_ids,
                "path_source_scope": "single_article" if len(article_ids) == 1 else "multi_article",
                "schema_version": SCHEMA_VERSION,
            }
        )
    return paths
