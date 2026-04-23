with open('verify_pipeline.py', 'r') as f:
    content = f.read()

content += """
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    test_pipeline()
"""

with open('verify_pipeline.py', 'w') as f:
    f.write(content)
