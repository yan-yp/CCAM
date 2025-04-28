from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models.matching_evalua import Matching
from models.utils import (compute_pose_error, compute_epipolar_error, estimate_pose_magsac,
                          estimate_pose_sgm, make_matching_plot,make_matching_plot_dgmc,
                          error_colormap, AverageTimer, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
from utils.common import make_grouping_plot
from models.metrics import pose_auc, approx_pose_auc, compute_epi_inlier
torch.set_grad_enabled(False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--viz_group', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')

    parser.add_argument(
        '--superglue', default='/home/leo/projects/SuperGlue_MegaDepth_clean/output/train/6_6_finetune/weights/last.pt',
        help='SuperGlue weights')
    # /media/leo/Datasets/openglue_output/SuperPointNet_960_720_cache__attn_softmax__laf_none__2022-05-22-20-07-47/superglue-step=1980000.ckpt\
    # /home/leo/projects/SuperGlue_MegaDepth_clean/models/weights/superglue_outdoor.pth
    # /media/leo/Projects/Luxiaoyong/SuperGlue_SuperPoint_MegaDepth-main/finetune_weight/best.pt
    # /home/leo/下载/best.pt
    # /home/seu/SuperGlue_SuperPoint_MegaDepth-main/SuperGlue_SuperPoint_MegaDepth-main/homo_pretrain_model/best.pt
    parser.add_argument(
        '--superpoint', default='/home/leo/projects/SuperGlue_MegaDepth_clean/models/weights/superpoint_v1.pth',
        help='SuperGlue weights')
    # /media/leo/Projects/Luxiaoyong/SuperGlue_SuperPoint_MegaDepth-main/models/weights/superpoint_v1.pth
    # /media/leo/Projects/Luxiaoyong/SuperGlue_training-main/output/final_model/superglue_conv_sync_wave_plus_1/weights/best.pt
    parser.add_argument(
        '--max_keypoints', type=int, default=2048,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=3,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=100,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.25,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1600],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true', default=True,
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization based on OpenCV instead of Matplotlib')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')

    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--pairs_list', type=str, default='assets/yfcc_test_pairs_with_gt_mini.txt',
        help='Path to the list of image pairs')
    #assets/yfcc_test_pairs_with_gt.txt
    #assets/scannet_sample_pairs_with_gt.txt
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')

    parser.add_argument(
        '--data_dir', type=str, default='/media/leo/Datasets/raw_data/yfcc100m',
        help='Path to the directory that contains the images')
    # D:\Luxiaoyong\OANet-master\raw_data_yfcc\raw_data\yfcc100m\
    parser.add_argument(
        '--results_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optional,'
             'visualizations are written')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.pairs_list, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.pairs_list))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'weights_path': opt.superpoint,
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights_path': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            'num_layers': 9,
            'use_dropout': False,
            'atten_dropout': 0.
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    data_dir = Path(opt.data_dir)
    print('Looking for data in directory \"{}\"'.format(data_dir))
    results_dir = Path(opt.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(results_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(results_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(results_dir))

    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = results_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = results_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = results_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = results_dir / '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)
        group_viz_path = results_dir / '{}_{}_group.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            data_dir / name0,  opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            data_dir / name1,  opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                data_dir/name0, data_dir/name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            for k, v in pred.items():
                if k != 'matches0' and k!='matches1' and k!='matching_scores0'and k != 'matching_scores1' and k != 'skip_train' and k != 'd0' and k != 'd1':
                    pred[k] = v[0]
            #pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'].cpu().numpy(), pred['keypoints1'].cpu().numpy()
            matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            # epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            # correct = epi_errs < 5e-3
            # num_correct = np.sum(correct)
            # precision = np.mean(correct) if len(correct) > 0 else 0
            # matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            err_R, err_t, inlier_mask = estimate_pose_sgm(T_0to1, mkpts0, mkpts1, K0, K1, thresh)
            num_correct = np.sum(inlier_mask)
            precision = np.mean(inlier_mask) if len(inlier_mask) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        # 'epipolar_errors': epi_errs
                        }
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'Ours',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))
            #d0 = pred['d0']
            #d1 = pred['d1']
            #d0 = d0.squeeze().cpu().detach().numpy()
            #d1 = d1.squeeze().cpu().detach().numpy()
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, stem0, stem1, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches')

            timer.update('viz_match')

        if do_viz_eval:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'Ours',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                stem0, stem1, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Relative Pose')

            timer.update('viz_eval')

        if opt.viz_group:
            attn0, attn1 = pred['attn0'].cpu().numpy(), pred['attn1'].cpu().numpy(),
            # a, b = (np.where(attn0 == 1)[1]+1)/8, (np.where(attn1 == 1)[1]+1)
            color0 = cm.jet(((np.where(attn0 == 1)[1])+1)/6)
            color1 = cm.jet(((np.where(attn1 == 1)[1])+1)/6)
            color = [color0, color1]
            make_grouping_plot(
                image0, image1, kpts0, kpts1, color, group_viz_path)
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = results_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [0, 5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        approx_auc=approx_pose_auc(pose_errors,thresholds)
        app_aucs = [100.*yy for yy in approx_auc]
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            app_aucs[0], app_aucs[1], app_aucs[2], prec, ms))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))