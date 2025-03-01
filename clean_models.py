"""
Utility script to manage model files
Can be run separately to clean up disk space by archiving or removing old model files
"""
import os
import sys
import argparse
import logging
from deriv_bot.utils.model_manager import ModelManager
from deriv_bot.monitor.logger import setup_logger

logger = setup_logger('model_cleanup')

def parse_args():
    parser = argparse.ArgumentParser(description='Model File Management Utility')
    parser.add_argument('--action', choices=['archive', 'clean', 'both', 'stats'], default='both',
                        help='Action to perform: archive (move old models to archive), clean (delete expired archives), '
                             'both (archive and clean), or stats (show statistics only)')
    parser.add_argument('--keep', type=int, default=5,
                        help='Number of most recent models to keep in the models directory')
    parser.add_argument('--days', type=int, default=30,
                        help='Keep archived models for this many days')
    parser.add_argument('--models-dir', default='models',
                        help='Directory containing model files')
    parser.add_argument('--archive-dir', default='model_archive',
                        help='Directory for archived model files')
    parser.add_argument('--model-type', default=None,
                        help='Filter by model type (e.g., short_term, long_term)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information about each file')
    return parser.parse_args()

def main():
    args = parse_args()

    print("=== Model File Management Utility ===")

    # Create model manager
    model_manager = ModelManager(
        models_dir=args.models_dir,
        archive_dir=args.archive_dir,
        max_models=args.keep
    )

    # Always show stats first
    stats = model_manager.get_model_size_stats()
    print("\nCurrent model storage statistics:")
    print(f"- Active models: {stats.get('active_models_count', 0)} files, {stats.get('active_models_size_mb', 0):.2f} MB")
    print(f"- Archived models: {stats.get('archived_models_count', 0)} files, {stats.get('archived_models_size_mb', 0):.2f} MB")
    print(f"- Total storage used: {stats.get('total_size_mb', 0):.2f} MB")

    if args.action == 'stats':
        # If only stats requested, exit here
        return

    # Perform requested actions
    if args.action in ['archive', 'both']:
        model_type_msg = f" for type '{args.model_type}'" if args.model_type else ""
        dry_run_msg = " (DRY RUN)" if args.dry_run else ""
        print(f"\nArchiving old models{model_type_msg}{dry_run_msg} (keeping {args.keep} most recent)...")

        if not args.dry_run:
            archived_count = model_manager.archive_old_models(model_type=args.model_type)
            print(f"Archived {archived_count} model files to {args.archive_dir}")
        else:
            # For dry run we need to list files that would be archived
            # We can implement this logic similar to cleanup dry run
            print("Dry run: Would archive files but no changes made")

    if args.action in ['clean', 'both']:
        dry_run_msg = " (DRY RUN)" if args.dry_run else ""
        print(f"\nCleaning up expired archives{dry_run_msg} (older than {args.days} days)...")
        deleted_count = model_manager.cleanup_archive(keep_days=args.days, dry_run=args.dry_run)

        if args.dry_run:
            print(f"Dry run: Would delete {deleted_count} expired archive files")
        else:
            print(f"Deleted {deleted_count} expired archive files")

    # Show updated statistics after operations
    if args.action in ['archive', 'clean', 'both'] and not args.dry_run:
        stats = model_manager.get_model_size_stats()
        print("\nUpdated model storage statistics:")
        print(f"- Active models: {stats.get('active_models_count', 0)} files, {stats.get('active_models_size_mb', 0):.2f} MB")
        print(f"- Archived models: {stats.get('archived_models_count', 0)} files, {stats.get('archived_models_size_mb', 0):.2f} MB")
        print(f"- Total storage used: {stats.get('total_size_mb', 0):.2f} MB")

    # If verbose, show detailed file listings
    if args.verbose:
        print("\nDetailed file listings:")

        # List active models
        active_models = glob.glob(os.path.join(args.models_dir, "*.h5"))
        active_models.sort(key=os.path.getmtime, reverse=True)

        if active_models:
            print("\nActive models:")
            for model in active_models:
                model_age = (datetime.datetime.now() - 
                            datetime.datetime.fromtimestamp(os.path.getmtime(model))).days
                model_size = os.path.getsize(model) / (1024 * 1024)  # size in MB
                print(f"- {os.path.basename(model)}: Age {model_age} days, Size {model_size:.2f} MB")

        # List archived models (limit to 20 most recent)
        archived_models = glob.glob(os.path.join(args.archive_dir, "*.h5"))
        archived_models.sort(key=os.path.getmtime, reverse=True)

        if archived_models:
            print("\nArchived models (20 most recent):")
            for model in archived_models[:20]:
                model_age = (datetime.datetime.now() - 
                            datetime.datetime.fromtimestamp(os.path.getmtime(model))).days
                model_size = os.path.getsize(model) / (1024 * 1024)  # size in MB
                print(f"- {os.path.basename(model)}: Age {model_age} days, Size {model_size:.2f} MB")

    print("\nDone!")

if __name__ == "__main__":
    # Import here to avoid import error when used in verbose mode
    if '--verbose' in sys.argv:
        import glob
        import datetime

    main()

def get_directory_size(path):
    """Get total size of a directory in bytes"""
    total_size = 0
    if not os.path.exists(path):
        return 0
        
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
            
    return total_size

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes == 0:
        return "0 B"
        
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
        
    return f"{size_bytes:.2f} {units[i]}"