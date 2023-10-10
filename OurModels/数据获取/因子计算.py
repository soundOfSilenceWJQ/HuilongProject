import os

from loguru import logger
from quant_base.core import parse_date
from quant_stock.core import StockDataCacheV2
from quant_stock.pipeline import *
from typing import Optional, Sequence

import pandas as pd
import quant_xpress_py as qx
from loguru import logger
from quant_dataflow.utils import factor_gen
from quant_stock.pipeline import PeriodDays, FinancialExtractor
from quant_stock.types import RefIndexData
# from index_enhance.utils import eval_qx_factors



__all__ = ['Alpha158_Factors']

EPS = 1e-7


@factor_gen
def Alpha1(CLOSE, OPEN):
    Alpha1 = qx.div(qx.sub(CLOSE, OPEN), OPEN)
    return Alpha1


# KLEN = ($high-$low)/$open
@factor_gen
def Alpha2(OPEN, HIGH, LOW):
    Alpha2 = qx.div(qx.sub(HIGH, LOW), OPEN)
    return Alpha2


# KMID2 = ($close-$open)/($high-$low+EPS)
@factor_gen
def Alpha3(CLOSE, OPEN, HIGH, LOW):
    Alpha3 = qx.div(qx.sub(CLOSE, OPEN), qx.sub(HIGH + EPS, LOW))
    return Alpha3


# KUP = ($high-maximum($open, $close))/$open
@factor_gen
def Alpha4(CLOSE, OPEN, HIGH):
    Alpha4 = qx.div(qx.sub(HIGH, qx.maximum(OPEN, CLOSE)), OPEN)
    return Alpha4


# KUP2 = ($high-maximum($open, $close))/($high-$low+EPS)
@factor_gen
def Alpha5(CLOSE, OPEN, HIGH, LOW):
    Alpha5 = qx.div(qx.sub(HIGH, qx.maximum(OPEN, CLOSE)), qx.sub((HIGH + EPS), LOW))
    return Alpha5


# KLOW = (minimum($open, $close)-$low)/$open
@factor_gen
def Alpha6(CLOSE, OPEN, LOW):
    Alpha6 = qx.div(qx.sub(qx.minimum(OPEN, CLOSE), LOW), OPEN)
    return Alpha6


# KLOW2 = (minimum($open, $close)-$low)/($high-$low+EPS)
@factor_gen
def Alpha7(CLOSE, OPEN, HIGH, LOW):
    Alpha7 = qx.div(qx.sub(qx.minimum(OPEN, CLOSE), LOW), qx.sub((HIGH + EPS), LOW))
    return Alpha7


# KSFT = (2*$close-$high-$low)/$open
@factor_gen
def Alpha8(CLOSE, OPEN, HIGH, LOW):
    Alpha8 = qx.div(qx.sub(qx.sub(2 * CLOSE, HIGH), LOW), OPEN)
    return Alpha8


# KSFT2 = (2*$close-$high-$low)/($high-$low+EPS)
@factor_gen
def Alpha9(CLOSE, HIGH, LOW):
    Alpha9 = qx.div(qx.sub(qx.sub(2 * CLOSE, HIGH), LOW), qx.sub(HIGH + EPS, LOW))
    return Alpha9


# OPEN0=$open/$close
@factor_gen
def Alpha10(CLOSE, OPEN):
    Alpha10 = qx.div(OPEN, CLOSE)
    return Alpha10


# HIGH0 = $high/$close
@factor_gen
def Alpha11(CLOSE, HIGH):
    Alpha11 = qx.div(HIGH, CLOSE)
    return Alpha11


# LOW0 = $low/$close
@factor_gen
def Alpha12(CLOSE, LOW):
    Alpha12 = qx.div(LOW, CLOSE)
    return Alpha12


# VWAP0 = $vwap/$close"
@factor_gen
def Alpha13(CLOSE, VWAP):
    Alpha13 = qx.div(VWAP, CLOSE)
    return Alpha13


# 5,10,20,30,60日
# ROC = "Ref($close, %d)/$close" % d for d in windows
@factor_gen
def Alpha14(CLOSE):
    Alpha14 = CLOSE[5] / CLOSE
    return Alpha14


@factor_gen
def Alpha15(CLOSE):
    Alpha15 = CLOSE[10] / CLOSE
    return Alpha15


@factor_gen
def Alpha16(CLOSE):
    Alpha16 = CLOSE[20] / CLOSE
    return Alpha16


@factor_gen
def Alpha17(CLOSE):
    Alpha17 = CLOSE[30] / CLOSE
    return Alpha17


@factor_gen
def Alpha18(CLOSE):
    Alpha18 = CLOSE[60] / CLOSE
    return Alpha18


# MA = "Mean($close, %d)/$close" % d for d in windows
@factor_gen
def Alpha19(CLOSE):
    Alpha19 = qx.ts_mean(CLOSE, 5) / CLOSE
    return Alpha19


@factor_gen
def Alpha20(CLOSE):
    Alpha20 = qx.ts_mean(CLOSE, 10) / CLOSE
    return Alpha20


@factor_gen
def Alpha21(CLOSE):
    Alpha21 = qx.ts_mean(CLOSE, 20) / CLOSE
    return Alpha21


@factor_gen
def Alpha22(CLOSE):
    Alpha22 = qx.ts_mean(CLOSE, 30) / CLOSE
    return Alpha22


@factor_gen
def Alpha23(CLOSE):
    Alpha23 = qx.ts_mean(CLOSE, 60) / CLOSE
    return Alpha23


# STD = "Std($close, %d)/$close" % d for d in windows
@factor_gen
def Alpha24(CLOSE):
    Alpha24 = qx.ts_std(CLOSE, 5) / CLOSE
    return Alpha24


@factor_gen
def Alpha25(CLOSE):
    Alpha25 = qx.ts_std(CLOSE, 10) / CLOSE
    return Alpha25


@factor_gen
def Alpha26(CLOSE):
    Alpha26 = qx.ts_std(CLOSE, 20) / CLOSE
    return Alpha26


@factor_gen
def Alpha27(CLOSE):
    Alpha27 = qx.ts_std(CLOSE, 30) / CLOSE
    return Alpha27


@factor_gen
def Alpha28(CLOSE):
    Alpha28 = qx.ts_std(CLOSE, 60) / CLOSE
    return Alpha28


# BETA = "Slope($close, %d)/$close" % d for d in windows
@factor_gen
def Alpha29(CLOSE):
    Alpha29 = -(qx.ft_get(qx.ts_linear_reg(CLOSE, 5, count_nan=False), 0) / CLOSE)
    return Alpha29


@factor_gen
def Alpha30(CLOSE):
    Alpha30 = -(qx.ft_get(qx.ts_linear_reg(CLOSE, 10, count_nan=False), 0) / CLOSE)
    return Alpha30


@factor_gen
def Alpha31(CLOSE):
    Alpha31 = -(qx.ft_get(qx.ts_linear_reg(CLOSE, 20, count_nan=False), 0) / CLOSE)
    return Alpha31


@factor_gen
def Alpha32(CLOSE):
    Alpha32 = -(qx.ft_get(qx.ts_linear_reg(CLOSE, 30, count_nan=False), 0) / CLOSE)
    return Alpha32


@factor_gen
def Alpha33(CLOSE):
    Alpha33 = -(qx.ft_get(qx.ts_linear_reg(CLOSE, 5, count_nan=False), 0) / CLOSE)
    return Alpha33


# RSQR = "Rsquare($close, %d)" % d for d in windows
@factor_gen
def Alpha34(CLOSE):
    # 计算SSR
    residuals = qx.ft_get(qx.ts_linear_residual(CLOSE, 5), 0)
    SSR = qx.ts_sum(residuals ** 2, 5)  # 对过去5个时刻的残差平方求和

    # 计算SST
    mean_CLOSE = qx.ts_mean(CLOSE, 5)
    SST = qx.ts_sum((CLOSE - mean_CLOSE) ** 2, 5)  # 对过去5个时刻的差异平方求和

    # 计算R^2
    Alpha34 = 1 - qx.div(SSR, SST)
    return Alpha34


@factor_gen
def Alpha35(CLOSE):
    # 计算SSR
    residuals = qx.ft_get(qx.ts_linear_residual(CLOSE, 10), 0)
    SSR = qx.ts_sum(residuals ** 2, 10)  # 对过去10个时刻的残差平方求和

    # 计算SST
    mean_CLOSE = qx.ts_mean(CLOSE, 10)
    SST = qx.ts_sum((CLOSE - mean_CLOSE) ** 2, 10)  # 对过去10个时刻的差异平方求和

    # 计算R^2
    Alpha35 = 1 - qx.div(SSR, SST)
    return Alpha35


@factor_gen
def Alpha36(CLOSE):
    # 计算SSR
    residuals = qx.ft_get(qx.ts_linear_residual(CLOSE, 20), 0)
    SSR = qx.ts_sum(residuals ** 2, 20)  # 对过去10个时刻的残差平方求和

    # 计算SST
    mean_CLOSE = qx.ts_mean(CLOSE, 20)
    SST = qx.ts_sum((CLOSE - mean_CLOSE) ** 2, 20)  # 对过去10个时刻的差异平方求和

    # 计算R^2
    Alpha36 = 1 - qx.div(SSR, SST)
    return Alpha36


@factor_gen
def Alpha37(CLOSE):
    # 计算SSR
    residuals = qx.ft_get(qx.ts_linear_residual(CLOSE, 30), 0)
    SSR = qx.ts_sum(residuals ** 2, 30)  # 对过去10个时刻的残差平方求和

    # 计算SST
    mean_CLOSE = qx.ts_mean(CLOSE, 30)
    SST = qx.ts_sum((CLOSE - mean_CLOSE) ** 2, 30)  # 对过去10个时刻的差异平方求和

    # 计算R^2
    Alpha37 = 1 - qx.div(SSR, SST)
    return Alpha37


@factor_gen
def Alpha38(CLOSE):
    # 计算SSR
    residuals = qx.ft_get(qx.ts_linear_residual(CLOSE, 60), 0)
    SSR = qx.ts_sum(residuals ** 2, 60)  # 对过去10个时刻的残差平方求和

    # 计算SST
    mean_CLOSE = qx.ts_mean(CLOSE, 60)
    SST = qx.ts_sum((CLOSE - mean_CLOSE) ** 2, 60)  # 对过去10个时刻的差异平方求和

    # 计算R^2
    Alpha38 = 1 - qx.div(SSR, SST)
    return Alpha38


# RESI = "Resi($close, %d)/$close" % d for d in windows
@factor_gen
def Alpha39(CLOSE):
    Alpha39 = qx.ft_get(qx.ts_linear_residual(CLOSE, 5), 0)
    return Alpha39


@factor_gen
def Alpha40(CLOSE):
    Alpha40 = qx.ft_get(qx.ts_linear_residual(CLOSE, 10), 0)
    return Alpha40


@factor_gen
def Alpha41(CLOSE):
    Alpha41 = qx.ft_get(qx.ts_linear_residual(CLOSE, 20), 0)
    return Alpha41


@factor_gen
def Alpha42(CLOSE):
    Alpha42 = qx.ft_get(qx.ts_linear_residual(CLOSE, 30), 0)
    return Alpha42


@factor_gen
def Alpha43(CLOSE):
    Alpha43 = qx.ft_get(qx.ts_linear_residual(CLOSE, 60), 0)
    return Alpha43


# MAX = "Max($high, %d)/$close" % d for d in windows
@factor_gen
def Alpha44(CLOSE, HIGH):
    Alpha44 = qx.div(qx.ts_max(HIGH, 5), CLOSE)
    return Alpha44


@factor_gen
def Alpha45(CLOSE, HIGH):
    Alpha45 = qx.div(qx.ts_max(HIGH, 10), CLOSE)
    return Alpha45


@factor_gen
def Alpha46(CLOSE, HIGH):
    Alpha46 = qx.div(qx.ts_max(HIGH, 20), CLOSE)
    return Alpha46


@factor_gen
def Alpha47(CLOSE, HIGH):
    Alpha47 = qx.div(qx.ts_max(HIGH, 30), CLOSE)
    return Alpha47


@factor_gen
def Alpha48(CLOSE, HIGH):
    Alpha48 = qx.div(qx.ts_max(HIGH, 60), CLOSE)
    return Alpha48


# MIN = "Min($low, %d)/$close" % d for d in windows
@factor_gen
def Alpha49(CLOSE, LOW):
    Alpha49 = qx.div(qx.ts_min(LOW, 5), CLOSE)
    return Alpha49


@factor_gen
def Alpha50(CLOSE, LOW):
    Alpha50 = qx.div(qx.ts_min(LOW, 10), CLOSE)
    return Alpha50


@factor_gen
def Alpha51(CLOSE, LOW):
    Alpha51 = qx.div(qx.ts_min(LOW, 20), CLOSE)
    return Alpha51


@factor_gen
def Alpha52(CLOSE, LOW):
    Alpha52 = qx.div(qx.ts_min(LOW, 30), CLOSE)
    return Alpha52


@factor_gen
def Alpha53(CLOSE, LOW):
    Alpha53 = qx.div(qx.ts_min(LOW, 60), CLOSE)
    return Alpha53


# QTLU = "Quantile($close, %d, 0.8)/$close" % d for d in windows
@factor_gen
def Alpha54(CLOSE):
    Alpha54 = qx.ts_quantile(CLOSE, 5, 0.8)
    return Alpha54


@factor_gen
def Alpha55(CLOSE):
    Alpha55 = qx.ts_quantile(CLOSE, 10, 0.8)
    return Alpha55


@factor_gen
def Alpha56(CLOSE):
    Alpha56 = qx.ts_quantile(CLOSE, 20, 0.8)
    return Alpha56


@factor_gen
def Alpha57(CLOSE):
    Alpha57 = qx.ts_quantile(CLOSE, 30, 0.8)
    return Alpha57


@factor_gen
def Alpha58(CLOSE):
    Alpha58 = qx.ts_quantile(CLOSE, 60, 0.8)
    return Alpha58


# QTLD = "Quantile($close, %d, 0.2)/$close" % d for d in windows
@factor_gen
def Alpha59(CLOSE):
    Alpha59 = qx.ts_quantile(CLOSE, 5, 0.2)
    return Alpha59


@factor_gen
def Alpha60(CLOSE):
    Alpha60 = qx.ts_quantile(CLOSE, 10, 0.2)
    return Alpha60


@factor_gen
def Alpha61(CLOSE):
    Alpha61 = qx.ts_quantile(CLOSE, 20, 0.2)
    return Alpha61


@factor_gen
def Alpha62(CLOSE):
    Alpha62 = qx.ts_quantile(CLOSE, 30, 0.2)
    return Alpha62


@factor_gen
def Alpha63(CLOSE):
    Alpha63 = qx.ts_quantile(CLOSE, 60, 0.2)
    return Alpha63


# RANK = "Rank($close, %d)" % d for d in windows
@factor_gen
def Alpha64(CLOSE):
    Alpha64 = qx.ts_rank_asc(CLOSE, 5)
    return Alpha64


@factor_gen
def Alpha65(CLOSE):
    Alpha65 = qx.ts_rank_asc(CLOSE, 10)
    return Alpha65


@factor_gen
def Alpha66(CLOSE):
    Alpha66 = qx.ts_rank_asc(CLOSE, 20)
    return Alpha66


@factor_gen
def Alpha67(CLOSE):
    Alpha67 = qx.ts_rank_asc(CLOSE, 30)
    return Alpha67


@factor_gen
def Alpha68(CLOSE):
    Alpha68 = qx.ts_rank_asc(CLOSE, 60)
    return Alpha68


# RSV = "($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+EPS)" % (d, d, d) for d in windows
@factor_gen
def Alpha69(CLOSE, HIGH, LOW):
    adv1 = qx.ts_min(LOW, 5)
    adv2 = qx.ts_max(HIGH, 5)
    Alpha69 = qx.div(CLOSE - adv1, adv2 + EPS - adv1)
    return Alpha69


@factor_gen
def Alpha70(CLOSE, HIGH, LOW):
    adv3 = qx.ts_min(LOW, 10)
    adv4 = qx.ts_max(HIGH, 10)
    Alpha70 = qx.div(CLOSE - adv3, adv4 + EPS - adv3)
    return Alpha70


@factor_gen
def Alpha71(CLOSE, HIGH, LOW):
    adv5 = qx.ts_min(LOW, 20)
    adv6 = qx.ts_max(HIGH, 20)
    Alpha71 = qx.div(CLOSE - adv5, adv6 + EPS - adv5)
    return Alpha71


@factor_gen
def Alpha72(CLOSE, HIGH, LOW):
    adv7 = qx.ts_min(LOW, 30)
    adv8 = qx.ts_max(HIGH, 30)
    Alpha72 = qx.div(CLOSE - adv7, adv8 + EPS - adv7)
    return Alpha72


@factor_gen
def Alpha73(CLOSE, HIGH, LOW):
    adv9 = qx.ts_min(LOW, 60)
    adv10 = qx.ts_max(HIGH, 60)
    Alpha73 = qx.div(CLOSE - adv9, adv10 + EPS - adv9)
    return Alpha73


# IMAX = "IdxMax($high, %d)/%d" % (d, d) for d in windows
@factor_gen
def Alpha74(HIGH):
    Alpha74 = (5 - qx.ts_argmax(HIGH, 5)) / 5
    return Alpha74


@factor_gen
def Alpha75(HIGH):
    Alpha75 = (10 - qx.ts_argmax(HIGH, 10)) / 10
    return Alpha75


@factor_gen
def Alpha76(HIGH):
    Alpha76 = (20 - qx.ts_argmax(HIGH, 20)) / 20
    return Alpha76


@factor_gen
def Alpha77(HIGH):
    Alpha77 = (30 - qx.ts_argmax(HIGH, 30)) / 30
    return Alpha77


@factor_gen
def Alpha78(HIGH):
    Alpha78 = (60 - qx.ts_argmax(HIGH, 60)) / 60
    return Alpha78


# IMIN = "IdxMin($low, %d)/%d" % (d, d) for d in windows
@factor_gen
def Alpha79(LOW):
    Alpha79 = (5 - qx.ts_argmin(LOW, 5)) / 5
    return Alpha79


@factor_gen
def Alpha80(LOW):
    Alpha80 = qx.ts_argmin(LOW, 10) / 10
    return Alpha80


@factor_gen
def Alpha81(LOW):
    Alpha81 = qx.ts_argmin(LOW, 20) / 20
    return Alpha81


@factor_gen
def Alpha82(LOW):
    Alpha82 = qx.ts_argmin(LOW, 30) / 30
    return Alpha82


@factor_gen
def Alpha83(LOW):
    Alpha83 = qx.ts_argmin(LOW, 60) / 60
    return Alpha83


# IMXD = "(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows
@factor_gen
def Alpha84(HIGH, LOW):
    Alpha84 = qx.sub((5 - qx.ts_argmax(HIGH, 5 + 1)) / 5, (5 - qx.ts_argmin(LOW, 5 + 1)) / 5) / 5
    return Alpha84


@factor_gen
def Alpha85(HIGH, LOW):
    Alpha85 = qx.sub((10 - qx.ts_argmax(HIGH, 10 + 1)) / 10, (10 - qx.ts_argmin(LOW, 10 + 1)) / 10) / 10
    return Alpha85


@factor_gen
def Alpha86(HIGH, LOW):
    Alpha86 = qx.sub((20 - qx.ts_argmax(HIGH, 20 + 1)) / 20, (20 - qx.ts_argmin(LOW, 20 + 1)) / 20) / 20
    return Alpha86


@factor_gen
def Alpha87(HIGH, LOW):
    Alpha87 = qx.sub((30 - qx.ts_argmax(HIGH, 30 + 1)) / 30, (30 - qx.ts_argmin(LOW, 30 + 1)) / 30) / 30
    return Alpha87


@factor_gen
def Alpha88(HIGH, LOW):
    Alpha88 = qx.sub((60 - qx.ts_argmax(HIGH, 60 + 1)) / 60, (60 - qx.ts_argmin(LOW, 60 + 1)) / 5) / 60
    return Alpha88


# CORR = "Corr($close, Log($volume+1), %d)" % d for d in windows
@factor_gen
def Alpha89(CLOSE, VOLUME):
    Alpha89 = qx.ts_corr(CLOSE, qx.log(VOLUME + 1), 5)
    return Alpha89


@factor_gen
def Alpha90(CLOSE, VOLUME):
    Alpha90 = qx.ts_corr(CLOSE, qx.log(VOLUME + 1), 10)
    return Alpha90


@factor_gen
def Alpha91(CLOSE, VOLUME):
    Alpha91 = qx.ts_corr(CLOSE, qx.log(VOLUME + 1), 20)
    return Alpha91


@factor_gen
def Alpha92(CLOSE, VOLUME):
    Alpha92 = qx.ts_corr(CLOSE, qx.log(VOLUME + 1), 30)
    return Alpha92


@factor_gen
def Alpha93(CLOSE, VOLUME):
    Alpha93 = qx.ts_corr(CLOSE, qx.log(VOLUME + 1), 60)
    return Alpha93


# CORD = "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows
@factor_gen
def Alpha94(CLOSE, VOLUME):
    Alpha94 = qx.ts_corr(CLOSE / CLOSE[1], qx.log((VOLUME / VOLUME[1]) + 1), 5)
    return Alpha94


@factor_gen
def Alpha95(CLOSE, VOLUME):
    Alpha95 = qx.ts_corr(CLOSE / CLOSE[1], qx.log((VOLUME / VOLUME[1]) + 1), 10)
    return Alpha95


@factor_gen
def Alpha96(CLOSE, VOLUME):
    Alpha96 = qx.ts_corr(CLOSE / CLOSE[1], qx.log((VOLUME / VOLUME[1]) + 1), 20)
    return Alpha96


@factor_gen
def Alpha97(CLOSE, VOLUME):
    Alpha97 = qx.ts_corr(CLOSE / CLOSE[1], qx.log((VOLUME / VOLUME[1]) + 1), 30)
    return Alpha97


@factor_gen
def Alpha98(CLOSE, VOLUME):
    Alpha98 = qx.ts_corr(CLOSE / CLOSE[1], qx.log((VOLUME / VOLUME[1]) + 1), 60)
    return Alpha98


# CNTP = "Mean($close>Ref($close, 1), %d)" % d for d in windows
@factor_gen
def Alpha99(CLOSE):
    Alpha99 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 5)
    return Alpha99


@factor_gen
def Alpha100(CLOSE):
    Alpha100 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 10)
    return Alpha100


@factor_gen
def Alpha101(CLOSE):
    Alpha101 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 20)
    return Alpha101


@factor_gen
def Alpha102(CLOSE):
    Alpha102 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 30)
    return Alpha102


@factor_gen
def Alpha103(CLOSE):
    Alpha103 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 60)
    return Alpha103


# CNTN = "Mean($close<Ref($close, 1), %d)" % d for d in windows
@factor_gen
def Alpha104(CLOSE):
    Alpha104 = qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 5)
    return Alpha104


@factor_gen
def Alpha105(CLOSE):
    Alpha105 = qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 10)
    return Alpha105


@factor_gen
def Alpha106(CLOSE):
    Alpha106 = qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 20)
    return Alpha106


@factor_gen
def Alpha107(CLOSE):
    Alpha107 = qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 30)
    return Alpha107


@factor_gen
def Alpha108(CLOSE):
    Alpha108 = qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 60)
    return Alpha108


# CNTD = "Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows
@factor_gen
def Alpha109(CLOSE):
    Alpha109 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 5) - qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 5)
    return Alpha109


@factor_gen
def Alpha110(CLOSE):
    Alpha110 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 10) - qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 10)
    return Alpha110


@factor_gen
def Alpha111(CLOSE):
    Alpha111 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 20) - qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 20)
    return Alpha111


@factor_gen
def Alpha112(CLOSE):
    Alpha112 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 30) - qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 30)
    return Alpha112


@factor_gen
def Alpha113(CLOSE):
    Alpha113 = qx.ts_mean(qx.maximum(CLOSE, CLOSE[1]), 60) - qx.ts_mean(qx.minimum(CLOSE, CLOSE[1]), 60)
    return Alpha113


# SUMP = "Sum(maximum($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+EPS)" % (d, d) for d in windows
@factor_gen
def Alpha114(CLOSE):
    Alpha114 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 5) / qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 5)
    return Alpha114


@factor_gen
def Alpha115(CLOSE):
    Alpha115 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 10) / qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 10)
    return Alpha115


@factor_gen
def Alpha116(CLOSE):
    Alpha116 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 20) / qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 20)
    return Alpha116


@factor_gen
def Alpha117(CLOSE):
    Alpha117 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 30) / qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 30)
    return Alpha117


@factor_gen
def Alpha118(CLOSE):
    Alpha118 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 60) / qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 60)
    return Alpha118


# SUMN =  "Sum(maximum(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+EPS)" % (d, d)for d in windows  # Can be derived from SUMP by SUMN = 1 - SUMP
@factor_gen
def Alpha119(CLOSE):
    Alpha119 = qx.ts_sum(qx.maximum(CLOSE[1] - 1, 0), 5) / (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 5) + EPS)
    return Alpha119


@factor_gen
def Alpha120(CLOSE):
    Alpha120 = qx.ts_sum(qx.maximum(CLOSE[1] - 1, 0), 10) / (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 10) + EPS)
    return Alpha120


@factor_gen
def Alpha121(CLOSE):
    Alpha121 = qx.ts_sum(qx.maximum(CLOSE[1] - 1, 0), 20) / (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 20) + EPS)
    return Alpha121


@factor_gen
def Alpha122(CLOSE):
    Alpha122 = qx.ts_sum(qx.maximum(CLOSE[1] - 1, 0), 30) / (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 30) + EPS)
    return Alpha122


@factor_gen
def Alpha123(CLOSE):
    Alpha123 = qx.ts_sum(qx.maximum(CLOSE[1] - 1, 0), 60) / (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 60) + EPS)
    return Alpha123


# SUMD =  "(Sum(maximum($close-Ref($close, 1), 0), %d)-Sum(maximum(Ref($close, 1)-$close, 0), %d))"
#                    "/(Sum(Abs($close-Ref($close, 1)), %d)+EPS)" % (d, d, d)
#                    for d in windows
@factor_gen
def Alpha124(CLOSE):
    ratio1 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 5) - qx.ts_sum(qx.maximum(CLOSE[1] - CLOSE, 0), 5)
    ratio2 = qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 5) + EPS
    Alpha124 = ratio1 / ratio2
    return Alpha124


@factor_gen
def Alpha125(CLOSE):
    ratio3 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 10) - qx.ts_sum(qx.maximum(CLOSE[1] - CLOSE, 0), 10)
    ratio4 = (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 10) + EPS)
    Alpha125 = ratio3 / ratio4
    return Alpha125


@factor_gen
def Alpha126(CLOSE):
    ratio5 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 20) - qx.ts_sum(qx.maximum(CLOSE[1] - CLOSE, 0), 20)
    ratio6 = (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 20) + EPS)
    Alpha126 = ratio5 / ratio6
    return Alpha126


@factor_gen
def Alpha127(CLOSE):
    ratio7 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 30) - qx.ts_sum(qx.maximum(CLOSE[1] - CLOSE, 0), 30)
    ratio8 = (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 30) + EPS)
    Alpha127 = ratio7 / ratio8
    return Alpha127


@factor_gen
def Alpha128(CLOSE):
    ratio9 = qx.ts_sum(qx.maximum(CLOSE - CLOSE[1], 0), 60) - qx.ts_sum(qx.maximum(CLOSE[1] - CLOSE, 0), 60)
    ratio10 = (qx.ts_sum(qx.abs(CLOSE - CLOSE[1]), 60) + EPS)
    Alpha128 = ratio9 / ratio10
    return Alpha128


# VMA = "Mean($volume, %d)/($volume+EPS)" % d for d in windows
@factor_gen
def Alpha129(VOLUME):
    Alpha129 = qx.ts_mean(VOLUME, 5) / (VOLUME + EPS)
    return Alpha129


@factor_gen
def Alpha130(VOLUME):
    Alpha130 = qx.ts_mean(VOLUME, 10) / (VOLUME + EPS)
    return Alpha130


@factor_gen
def Alpha131(VOLUME):
    Alpha131 = qx.ts_mean(VOLUME, 20) / (VOLUME + EPS)
    return Alpha131


@factor_gen
def Alpha132(VOLUME):
    Alpha132 = qx.ts_mean(VOLUME, 30) / (VOLUME + EPS)
    return Alpha132


@factor_gen
def Alpha133(VOLUME):
    Alpha133 = qx.ts_mean(VOLUME, 60) / (VOLUME + EPS)
    return Alpha133


# VSTD = "Std($volume, %d)/($volume+EPS)" % d for d in windows
@factor_gen
def Alpha134(VOLUME):
    Alpha134 = qx.ts_std(VOLUME, 5) / (VOLUME + EPS)
    return Alpha134


@factor_gen
def Alpha135(VOLUME):
    Alpha135 = qx.ts_std(VOLUME, 10) / (VOLUME + EPS)
    return Alpha135


@factor_gen
def Alpha136(VOLUME):
    Alpha136 = qx.ts_std(VOLUME, 20) / (VOLUME + EPS)
    return Alpha136


@factor_gen
def Alpha137(VOLUME):
    Alpha137 = qx.ts_std(VOLUME, 30) / (VOLUME + EPS)
    return Alpha137


@factor_gen
def Alpha138(VOLUME):
    Alpha138 = qx.ts_std(VOLUME, 60) / (VOLUME + EPS)
    return Alpha138


# WVMA =   "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+EPS)" % (d, d) for d in windows
@factor_gen
def Alpha139(CLOSE, VOLUME):
    Alpha139 = (
        qx.ts_std(qx.abs(CLOSE / CLOSE[1] - 1) * VOLUME, 5)
        / (qx.ts_mean(abs(CLOSE / CLOSE[1] - 1) * VOLUME, 5) + EPS)
    )
    return Alpha139


@factor_gen
def Alpha140(CLOSE, VOLUME):
    Alpha140 = (
        qx.ts_std(qx.abs(CLOSE / CLOSE[1] - 1) * VOLUME, 10)
        / (qx.ts_mean(abs(CLOSE / CLOSE[1] - 1) * VOLUME, 10) + EPS)
    )
    return Alpha140


@factor_gen
def Alpha141(CLOSE, VOLUME):
    Alpha141 = (
        qx.ts_std(qx.abs(CLOSE / CLOSE[1] - 1) * VOLUME, 20)
        / (qx.ts_mean(abs(CLOSE / CLOSE[1] - 1) * VOLUME, 20) + EPS)
    )
    return Alpha141


@factor_gen
def Alpha142(CLOSE, VOLUME):
    Alpha142 = (
        qx.ts_std(qx.abs(CLOSE / CLOSE[1] - 1) * VOLUME, 30)
        / (qx.ts_mean(abs(CLOSE / CLOSE[1] - 1) * VOLUME, 30) + EPS)
    )
    return Alpha142


@factor_gen
def Alpha143(CLOSE, VOLUME):
    Alpha143 = (
        qx.ts_std(qx.abs(CLOSE / CLOSE[1] - 1) * VOLUME, 60)
        / (qx.ts_mean(abs(CLOSE / CLOSE[1] - 1) * VOLUME, 60) + EPS)
    )
    return Alpha143


# VSUMP = "Sum(maximum($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+EPS)" % (d, d) for d in windows
@factor_gen
def Alpha144(VOLUME):
    Alpha144 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 5) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 5) + EPS)
    return Alpha144


@factor_gen
def Alpha145(VOLUME):
    Alpha145 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 10) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 10) + EPS)
    return Alpha145


@factor_gen
def Alpha146(VOLUME):
    Alpha146 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 20) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 20) + EPS)
    return Alpha146


@factor_gen
def Alpha147(VOLUME):
    Alpha147 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 30) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 30) + EPS)
    return Alpha147


@factor_gen
def Alpha148(VOLUME):
    Alpha148 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 60) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 60) + EPS)
    return Alpha148


# VSUMN=  "Sum(maximum(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+EPS)" % (d, d) for d in windows
# Can be derived from VSUMP by VSUMN = 1 - VSUMP
@factor_gen
def Alpha149(VOLUME):
    Alpha149 = qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 5) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 5) + EPS)
    return Alpha149


@factor_gen
def Alpha150(VOLUME):
    Alpha150 = qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 10) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 10) + EPS)
    return Alpha150


@factor_gen
def Alpha151(VOLUME):
    Alpha151 = qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 20) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 20) + EPS)
    return Alpha151


@factor_gen
def Alpha152(VOLUME):
    Alpha152 = qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 30) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 30) + EPS)
    return Alpha152


@factor_gen
def Alpha153(VOLUME):
    Alpha153 = qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 60) / (qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 60) + EPS)
    return Alpha153


# VSUMD = "(Sum(maximum($volume-Ref($volume, 1), 0), %d)-Sum(maximum(Ref($volume, 1)-$volume, 0), %d))"
#                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+EPS)" % (d, d, d)
#                    for d in windows

@factor_gen
def Alpha154(VOLUME):
    sum1 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 5) - qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 5)
    sum2 = qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 5) + EPS
    Alpha154 = sum1 / sum2
    return Alpha154


@factor_gen
def Alpha155(VOLUME):
    sum3 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 10) - qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 10)
    sum4 = qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 10) + EPS
    Alpha155 = sum3 / sum4
    return Alpha155


@factor_gen
def Alpha156(VOLUME):
    sum5 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 20) - qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 20)
    sum6 = qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 20) + EPS
    Alpha156 = sum5 / sum6
    return Alpha156


@factor_gen
def Alpha157(VOLUME):
    sum7 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 30) - qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 30)
    sum8 = qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 30) + EPS
    Alpha157 = sum7 / sum8
    return Alpha157


@factor_gen
def Alpha158(VOLUME):
    sum9 = qx.ts_sum(qx.maximum(VOLUME - VOLUME[1], 0), 60) - qx.ts_sum(qx.maximum(VOLUME[1] - VOLUME, 0), 60)
    sum10 = qx.ts_sum(qx.abs(VOLUME - VOLUME[1]), 60) + EPS
    Alpha158 = sum9 / sum10
    return Alpha158

@factor_gen
def LNCAP(MARKET_CAP):
    LNCAP = qx.log(MARKET_CAP)
    return LNCAP


def Alpha158_Factors(df: pd.DataFrame,
                     financial: FinancialExtractor,
                     period: PeriodDays,
                     ) -> pd.DataFrame:
    """
    计算 Alpha158 因子。

    :param df: 股票的日 K 数据。
    :param financial: 股票的财报数据提取器。
    :param period: 换仓日对象。
    """
    logger.info('计算 Alpha158 因子 ...')

    g_dict = globals()
    fac_generators = {}
    for i in range(1, 159):  # Loop through values 1 to 158
        # Try to get the factor function from the global dictionary.
        # This way, even if some of the factor functions are missing, the loop won't break.
        fac_func = g_dict.get(f'Alpha{i}')
        if fac_func:
            fac_generators[f'Alpha158_{i:03d}'] = fac_func

        # Add the LNCAP function to fac_generators
    fac_generators['LNCAP'] = g_dict.get('LNCAP')

    df = eval_qx_factors(df, fac_generators, period_days=period)
    return df


if __name__ == '__main__':
    # 回测日期范围
    start_date = parse_date('2009-06-01')
    end_date = parse_date('2023-06-30')
    period_days = 'monthly'

    # 计算因子
    if not os.path.isfile('./_factor_origin_statues.hdf'):
        data, loader = load_factors(
            start_date=start_date,
            end_date=end_date,
            period_days=period_days,
            fac_loaders=[
                Alpha158_Factors,
            ],
            cache=StockDataCacheV2('./_cache'),
            ref_index='沪深300',
        )
        logger.info('因子计算完成。')
        data.to_hdf('./_factor_origin_statues.hdf', key='data')
        logger.info('因子已存储到文件。')
