
from runx.logx import logx

from formatter.sentence.mse_formatter import mse_formatter
from formatter.sentence.mse_seprate_formatter import mse_seprate_formatter
from formatter.sentence.teacher_sim_formatter import teacher_sim_formatter
from utils.message_utils import gen_time_str, warning_msg, infor_msg, erro_msg, epoch_msg, correct_msg

formatter_list = {
    "mse_formatter": mse_formatter,
    "teacher_sim_formatter":teacher_sim_formatter,
    "mse_seprate_formatter": mse_seprate_formatter
}


def init_formatter(config, model, *args, **params):

    which = config.get("data", "train_formatter")

    if which in formatter_list:
        formatter = formatter_list[which](config, model, *args, **params)

        return formatter
    else:
        logx.msg(erro_msg(f"There is no formatter called {which}, check your config."))
        raise NotImplementedError
