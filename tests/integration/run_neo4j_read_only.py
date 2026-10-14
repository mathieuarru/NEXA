#!/usr/bin/env python3
"""Build the graph Dockerfiles on tiny fixtures and check the graphs stay read-only.

Requires Docker. Creates and removes only UUID-named test containers, volumes
and images; existing graphs are never touched. No LLM, PostgreSQL or real KG.

Usage: poetry run python tests/integration/run_neo4j_read_only.py [--graphs ...] [--output FILE]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from neo4j import GraphDatabase, exceptions
from nexa.knowledge_graph.neo4j_handler import Neo4jHandler
from nexa.knowledge_graph.read_only import POLICY_ERROR_CODE
from primekg.prepare_primekg import REQUIRED_COLUMNS


def docker(*args, check=True):
    """Run a docker command; raise on a non-zero exit when `check`."""
    result = subprocess.run(["docker", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode:
        raise RuntimeError(f"docker {' '.join(args)} failed:\n{result.stdout[-12000:]}")
    return result


def fixtures(context):
    """Copy the build files into `context` with a synthetic fixture for each graph.

    Each fixture has the format the image downloads and sits where the build
    looks for a local copy. Returns the build args pinning each fixture's MD5,
    so no build needs the network.
    """
    for source in ["docker", "alzkb/load_alzkb121.sh", "alzkb/cypherl_to_json.py",
                   "hetionet/load_hetionet.sh", "primekg/load_primekg.sh", "primekg/prepare_primekg.py",
                   "Dockerfile.alzkb121", "Dockerfile.hetionet", "Dockerfile.primekg", "Dockerfile.primekg.dockerignore"]:
        src, dst = ROOT / source, context / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    md5 = lambda path: hashlib.md5(path.read_bytes()).hexdigest()
    # AlzKB: the Memgraph CYPHERL of the release, converted during the build.
    (context / "alzkb/source").mkdir()
    cypherl = context / "alzkb/source/alzkb-v1-2-1_memgraph"
    cypherl.write_text(
        "CREATE INDEX ON :__mg_vertex__(__mg_id__);\n"
        'CREATE (:__mg_vertex__:`Drug` {__mg_id__: 1, `name`: "Synthetic drug"});\n'
        'CREATE (:__mg_vertex__:`Disease` {__mg_id__: 2, `name`: "Synthetic disease"});\n'
        "MATCH (u:__mg_vertex__), (v:__mg_vertex__) WHERE u.__mg_id__ = 1 AND v.__mg_id__ = 2 "
        'CREATE (u)-[:`INDICATION` {`evidence`: "fixture"}]->(v);\n'
        "DROP INDEX ON :__mg_vertex__(__mg_id__);\n"
        "MATCH (u) REMOVE u:__mg_vertex__, u.__mg_id__;\n"
    )
    # Hetionet: a CSV listed in checksums.tsv (its loader here is the fixture
    # Cypher below, which does not read it).
    (context / "hetionet/data").mkdir()
    csv_fixture = context / "hetionet/data/node_Fixture.csv"
    csv_fixture.write_text("id,name\n1,Synthetic fixture only\n")
    (context / "hetionet/checksums.tsv").write_text(
        f"file\tgz_md5\tcsv_md5\n{csv_fixture.name}\t{'0' * 32}\t{md5(csv_fixture)}\n")
    (context / "hetionet/cypher").mkdir()
    (context / "hetionet/cypher/node_fixture.cypher").write_text(
        "CREATE (:Drug {name:'Synthetic drug'}), (:Disease {name:'Synthetic disease'});\n"
        "CREATE CONSTRAINT fixture_name IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE;\n"
    )
    (context / "hetionet/cypher/edge_fixture.cypher").write_text(
        "MATCH (d:Drug), (c:Disease) CREATE (d)-[:INDICATION {evidence:'fixture'}]->(c);\n"
    )
    (context / "primekg/data").mkdir()
    kg = context / "primekg/data/kg.csv"
    with kg.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_COLUMNS)
        writer.writerow(["indication", "indication", 1, "drug-1", "drug", "Synthetic drug", "fixture",
                         2, "disease-1", "disease", "Synthetic disease", "fixture"])
    return {"alzkb": [f"ALZKB_MD5={md5(cypherl)}"], "hetionet": [], "primekg": [f"PRIMEKG_MD5={md5(kg)}"]}


def raw(driver, query):
    with driver.session(database="neo4j") as session:
        return session.run(query).data()


def snapshot(driver):
    return {
        "nodes": raw(driver, "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties ORDER BY elementId(n)"),
        "relationships": raw(driver, "MATCH (a)-[r]->(b) RETURN a.name AS source, b.name AS target, type(r) AS type, properties(r) AS properties ORDER BY elementId(r)"),
    }


def expect_server_refusal(driver, query):
    """Assert that the server itself rejects `query`."""
    try:
        raw(driver, query)
    except exceptions.Neo4jError as e:
        code = e.code or ""
        assert any(s in code for s in ("ReadOnly", "Forbidden", "ProcedureNotFound")), (query, code, str(e))
        return {"query": query, "code": code}
    raise AssertionError(f"Server allowed forbidden operation: {query}")


def wait_ready(name, port):
    """Wait for Bolt on `port`; return a driver once the database reports read-only."""
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        state = json.loads(docker("inspect", name, "--format", "{{json .State}}").stdout)
        if not state["Running"]:
            raise AssertionError(f"Test container exited: {state}")
        driver = GraphDatabase.driver(f"bolt://127.0.0.1:{port}", auth=("neo4j", "test-password"),
                                     connection_timeout=1, connection_acquisition_timeout=2)
        try:
            # Check on the first successful connection: the graph must never
            # be reachable while writable.
            rows = raw(driver, "SHOW DATABASES YIELD name, access WHERE name = 'neo4j' RETURN access")
            assert rows == [{"access": "read-only"}], rows
            return driver
        except (exceptions.ServiceUnavailable, exceptions.SessionExpired):
            driver.close()
            time.sleep(1)
        except Exception:
            driver.close()
            raise
    raise TimeoutError("Test container did not become ready within 300 seconds.")


def check_graph(driver, port):
    """Check that reads work and writes are refused by NEXA and by the server."""
    baseline = snapshot(driver)
    assert len(baseline["nodes"]) == 2 and len(baseline["relationships"]) == 1, baseline
    handler = Neo4jHandler(port=str(port), password="test-password")
    read_queries = [
        "MATCH (n) RETURN n.name AS name ORDER BY name",
        "MATCH (n) RETURN COUNT(n) AS count",
        "CALL { MATCH (n) RETURN n.name AS name } RETURN name",
        "RETURN apoc.text.replace('A-B', '-', ' ') AS name",
        "CALL db.labels() YIELD label RETURN label",
        "CALL apoc.meta.stats() YIELD nodeCount RETURN nodeCount",
        "CALL apoc.meta.data() YIELD label RETURN label",
    ]
    blocked = [
        "MATCH (n) DETACH DELETE n", "CREATE (:Unexpected)",
        "MERGE (:Unexpected)", "MATCH (n) SET n.name = 'changed'",
        "MATCH (n) REMOVE n.name", "CALL { MATCH (n) DELETE n } RETURN 1",
        "CALL apoc.create.node(['Unexpected'], {})",
        "CALL apoc.cypher.doIt('CREATE (:Unexpected)', {})",
        "CALL dbms.setConfigValue('server.databases.read_only', '')",
        "CREATE INDEX unexpected_index FOR (n:Drug) ON (n.name)",
    ]
    try:
        for query in read_queries:
            data, warnings = handler.execute_query(query)
            assert data is not None, (query, warnings)
        elements, warnings = handler.execute_query_as_elements("MATCH (a)-[r]->(b) RETURN a,r,b")
        assert len(elements["nodes"]) == 2 and len(elements["edges"]) == 1, (elements, warnings)
        assert handler.execute_query("RETURN $text AS text", {"text": "DELETE; CREATE ()"})[0] == [{"text": "DELETE; CREATE ()"}]
        assert handler.get_nodes() and handler.get_relationships()
        assert handler.get_relationships_properties() is not None
        assert handler.schema_fingerprint()
        for query in blocked:
            for method in (handler.execute_query, handler.execute_query_as_elements):
                data, warnings = method(query)
                assert data is None and warnings[0]["code"] == POLICY_ERROR_CODE, (query, warnings)
        # Bypass NEXA entirely: server protection must still reject these.
        server_refusals = [expect_server_refusal(driver, query) for query in blocked]
        assert snapshot(driver) == baseline, "Node/relationship properties changed."
        return {"read_queries": read_queries, "application_refusals": len(blocked) * 2,
                "server_refusals": server_refusals, "contents_unchanged": True}
    finally:
        handler.close()


def check_image(graph, context, build_args):
    """Build one graph image and check it stays read-only after restart and recreation."""
    token = uuid.uuid4().hex[:12]
    name = f"nexa-readonly-test-{graph}-{token}"
    image = f"{name}:test"
    volume = f"{name}-data"
    dockerfile = "Dockerfile.alzkb121" if graph == "alzkb" else f"Dockerfile.{graph}"
    print(f"[{graph}] Building {dockerfile} with a synthetic fixture...", flush=True)
    driver = None
    try:
        args = ["build", "--network=none", "-f", str(context / dockerfile), "-t", image]
        for arg in build_args[graph]:
            args.extend(["--build-arg", arg])
        docker(*args, str(context))
        docker("volume", "create", volume)

        def start():
            docker("run", "-d", "--name", name, "--memory=1g", "--cpus=1",
                   "-p", "127.0.0.1::7687", "-v", f"{volume}:/data",
                   "-e", "NEO4J_AUTH=neo4j/test-password",
                   "-e", "NEO4J_server_memory_heap_initial__size=256M",
                   "-e", "NEO4J_server_memory_heap_max__size=256M",
                   "-e", "NEO4J_server_memory_pagecache_size=64M", image)
            port = int(docker("port", name, "7687").stdout.strip().rsplit(":", 1)[1])
            print(f"[{graph}] Waiting for {name} on test port {port}...", flush=True)
            return wait_ready(name, port), port

        driver, port = start()
        print(f"[{graph}] Import succeeded; checking reads and rejected writes...", flush=True)
        evidence = check_graph(driver, port)
        baseline = snapshot(driver)
        docker("exec", name, "/opt/nexa/neo4j_healthcheck.sh")
        driver.close()
        driver = None
        print(f"[{graph}] Restarting the test container...", flush=True)
        docker("restart", "--time", "30", name)
        # Docker may allocate a different random host port after restart.
        port = int(docker("port", name, "7687").stdout.strip().rsplit(":", 1)[1])
        driver = wait_ready(name, port)
        expect_server_refusal(driver, "MATCH (n) DETACH DELETE n")
        assert snapshot(driver) == baseline
        driver.close()
        driver = None
        print(f"[{graph}] Recreating the test container on its imported volume...", flush=True)
        docker("rm", "-f", "-v", name)
        driver, port = start()
        expect_server_refusal(driver, "MATCH (n) SET n.name = 'changed'")
        assert snapshot(driver) == baseline
        evidence.update(restart_read_only=True, recreate_read_only=True, image_id=docker("inspect", name, "--format", "{{.Image}}").stdout.strip())
        print(f"[{graph}] All integration checks passed.", flush=True)
        return evidence
    except Exception:
        print(docker("logs", "--tail", "100", name, check=False).stdout, flush=True)
        raise
    finally:
        if driver is not None:
            driver.close()
        docker("rm", "-f", "-v", name, check=False)
        docker("volume", "rm", volume, check=False)
        docker("image", "rm", image, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", choices=["alzkb", "hetionet", "primekg"], default=["alzkb", "hetionet", "primekg"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"graphs": {}, "fixture": "2 synthetic nodes and 1 relationship", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with tempfile.TemporaryDirectory(prefix="nexa-readonly-fixtures-") as temp:
        context = Path(temp)
        build_args = fixtures(context)
        for graph in args.graphs:
            report["graphs"][graph] = check_image(graph, context, build_args)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
