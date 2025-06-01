import argparse
from task_logger import log_task, add_task
from suggester import suggest_task

def main():
    parser = argparse.ArgumentParser(description="RhythmTask AI Assistant")
    parser.add_argument("--log", type=str, help="Log a completed task")
    parser.add_argument("--add", type=str, help="Add a new task to the list")
    parser.add_argument("--suggest", action="store_true", help="Suggest a task based on your rhythm")
    args = parser.parse_args()

    if args.log:
        log_task(args.log)
        print(f"✅ Logged task: {args.log}")
    elif args.add:
        add_task(args.add)
        print(f"✅ Added new task: {args.add}")
    elif args.suggest:
       suggestion = suggest_task()
       print("DEBUG suggestion:", suggestion)
       print("\n✨ Suggested task to do now:")
       print(f"- {suggestion}")


    else:
        parser.print_help()
if __name__ == "__main__":
    main()

