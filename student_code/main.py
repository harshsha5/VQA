import argparse
from student_code.simple_baseline_experiment_runner import SimpleBaselineExperimentRunner
from student_code.coattention_experiment_runner import CoattentionNetExperimentRunner
from tensorboardX import SummaryWriter
import time
import torch


if __name__ == "__main__":
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, choices=['simple', 'coattention'], default='simple')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--use_saved_dictionaries', type=int, default=0)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--cache_location', type=str, default="")
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log_validation', action='store_true')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eval_model', action='store_true', default=False,
                        help='Evaluate Model instead of Training')
    parser.add_argument('--model_eval_path', type=str)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == "simple":
        experiment_runner_class = SimpleBaselineExperimentRunner
    elif args.model == "coattention":
        experiment_runner_class = CoattentionNetExperimentRunner
    else:
        raise ModuleNotFoundError()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_path = "runs/"+timestr+'/'
    writer = SummaryWriter(save_path)

    experiment_runner = experiment_runner_class(train_image_dir=args.train_image_dir,
                                                train_question_path=args.train_question_path,
                                                train_annotation_path=args.train_annotation_path,
                                                test_image_dir=args.test_image_dir,
                                                test_question_path=args.test_question_path,
                                                test_annotation_path=args.test_annotation_path,
                                                batch_size=args.batch_size,
                                                num_epochs=args.num_epochs,
                                                num_data_loader_workers=args.num_data_loader_workers,
                                                cache_location=args.cache_location,
                                                lr=args.lr,
                                                log_validation=args.log_validation,
                                                writer=writer,
                                                use_saved_dictionaries=args.use_saved_dictionaries,
                                                device=device)

    if(args.eval_model is False):
        experiment_runner.train()
    else:
        print("Evaluating model")
        experiment_runner._model.load_state_dict(torch.load(args.model_eval_path))
        experiment_runner._model.eval()
        experiment_runner.validate(current_step = 0)
        
    writer.close()
