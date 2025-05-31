# main.py
import argparse
from task_logger import log_task
from suggester import suggest_task

def main():
    parser = argparse.ArgumentParser(description="RhythmTask AI Assistant")
    parser.add_argument("--log", help="Log a completed task")
    parser.add_argument("--suggest", action="store_true", help="Suggest a task based on your rhythm")

    args = parser.parse_args()

    if args.log:
        log_task(args.log)
        print(f"✅ Logged task: {args.log}")

    elif args.suggest:
        suggestion = suggest_task()
        print("\n✨ Suggested tasks to do now:")
        for task in suggestion:
            print(f"- {task}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
