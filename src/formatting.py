"""Pretty-printing helpers for RAG results."""


def print_rag_result(result):
    width = 80

    print("=" * width)
    print("QUESTION")
    print("=" * width)
    print(result["question"])

    print("\n" + "=" * width)
    print("ANSWER")
    print("=" * width)
    print(result["answer"])

    if result["summary"]:
        print("\n" + "=" * width)
        print("SUMMARY")
        print("=" * width)
        print(result["summary"])

    print("\n" + "=" * width)
    print("SOURCES")
    print("=" * width)
    for source in result["sources"]:
        print(f"  [{source['source']}]  Page {source['page']}  |  Score: {source['score']}")
        print(f"  Preview: {source['preview']}")
        print("-" * width)
