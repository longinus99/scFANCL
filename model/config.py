import argparse

parser = argparse.ArgumentParser(description='ScCCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='10X_PBMC')
parser.add_argument('--cuda', type=int, default=0 , help='GPU id to use.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--select_gene', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--m', type=float, default=0.5)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--enc_1', type=int, default=200)
parser.add_argument('--enc_2', type=int, default=40)
parser.add_argument('--enc_3', type=int, default=60)
parser.add_argument('--mlp_dim', type=int, default=40)
parser.add_argument('--cluster_methods', type=str, default="KMeans")
parser.add_argument('--phi', type=float, default=0.95,
                    help='Threshold for false negative filtering in FN_InstanceLoss')

args = parser.parse_args()