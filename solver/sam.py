import math
import torch
import torch.optim
import numpy as np
import functools
from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class SAM(torch.optim.Optimizer):
    @configurable()
    def __init__(self, params, base_optimizer, rho, sam_variant="sam", asam_eta=0) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        self.sam_variant = sam_variant
        self.asam_eta = asam_eta
        super(SAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
        
    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sam_variant": args.sam_variant,
            "asam_eta": args.asam_eta,
        }
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self.sam_variant in ["sam", "asam"]:
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    if self.sam_variant == "asam":
                        e_w = group["rho"] / (grad_norm + 1e-7) * torch.square(self.asam_eta+torch.abs(p))*p.grad
                    elif self.sam_variant == "sam":
                        e_w = group["rho"] / (grad_norm + 1e-7) * p.grad      
                    self.state[p]["e_w"] = e_w
                    p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        
        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if self.sam_variant == "asam":
            norm = torch.norm(
                        torch.stack([
                            torch.norm((self.asam_eta + torch.abs(p))*p.grad, p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                   )
        elif self.sam_variant == "sam":
            norm = torch.norm(
                        torch.stack([
                            p.grad.norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                   )
        return norm

def _sample_hypersphere(shapes, device):
    sum_squares = torch.tensor(0.).to(torch.float64).to(device)
    arrays = []
    for s in shapes:
        x = torch.normal(0, 1, size=s).to(device)
        sum_squares += torch.sum(torch.pow(x, 2.))
        arrays.append(x)
    scale = 1./torch.sqrt(sum_squares)
    return arrays

def _get_torch_uniform(low, high, size, device):
    return ((high - low) * torch.rand(*size) + low).to(device)

def _sample_gaussian(shapes, device):
    arrays = []
    for s in shapes:
        x = torch.normal(0, 1, size=s).to(device)
        arrays.append(x)
    return arrays


def _sample_lebesgue(shapes, device, half_cube_len=1, seed=None):
    arrays = []
    for s in shapes:
        x = _get_torch_uniform(low=-1*half_cube_len, high=half_cube_len, size=s, device=device)
        arrays.append(x)
    return arrays


@OPTIMIZER_REGISTRY.register()
class ISAM(torch.optim.Optimizer):
    # args should have "isam_rho", "isam_n_samples" (number of samples for sharpness estimate), "isam_setting" (which will determine psi, phi, and mu)
    # see defaulf_cfg.py
    
    @configurable()
    def __init__(self, params, base_optimizer, rho, psi, psi_p, phi_p, n_samples, sample_fn, lam) -> None:
        # psi, psi_p, phi_p are functions from scalar to scalar. psi_p = psi prime, phi_p = phi prime
        # n_samples: number of samples to draw from mu
        # sample_fn: a function: [s1, s2, ..] where s's are shape tuples |--> [x1, x2, ...] where x_i is a tensor of shape s_i (samples from mu)
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.psi = psi
        self.psi_p = psi_p
        self.phi_p = phi_p
        self.n_samples = n_samples
        self.sample_fn = sample_fn
        self.lam = lam
        super(ISAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
    
    @classmethod
    def from_config(cls, args):
        d = {
            "rho": args.isam_rho,
            "n_samples": args.isam_n_samples,
            "lam": args.isam_lam,
        }
        if args.isam_setting == "trace":
            psi = lambda x: x
            psi_p = lambda x: 1.
            phi_p = lambda x: 1.
            sample_fn = _sample_hypersphere
        elif args.isam_setting == "det":
            psi = lambda x: torch.exp(-x/2.)
            psi_p = lambda x: -0.5*psi(x)
            # NOTE: (2pi)^d missing below, compensate with isam_lam term
            phi_p = lambda x: -2/torch.pow(x, 3)
            sample_fn = functools.partial(_sample_lebesgue, half_cube_len=args.isam_half_cube_len)
        else:
            raise ValueError(f"unsupported setting: {args.isam_setting}")
        
        d["psi"] = psi
        d["psi_p"] = psi_p
        d["phi_p"] = phi_p
        d["sample_fn"] = sample_fn
        return d
    
    @torch.no_grad()
    def set_current_gradients(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g0"] = p.grad
                self.state[p]["g_sum"] = torch.zeros_like(p.grad, requires_grad=False).to(p)

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            valid_ps = [p for p in group["params"] if p.grad is not None]
            for p in valid_ps:
                s = self.sample_fn([tuple(p.grad.shape)], p.device)[0]
                e_w = group["rho"] * s
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self, scale):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g_sum"] += scale*(p.grad - self.state[p]["g0"])
                p.sub_(self.state[p]["e_w"])

    @torch.no_grad()
    def final_step(self, scale):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.set_(self.state[p]["g0"] + self.lam*scale*self.state[p]["g_sum"])
        self.base_optimizer.step()

    
    @torch.no_grad()
    def step(self, loss, closure=None, **kwargs):
        assert closure is not None, "ISAM requires closure, which is not provided."

        self.set_current_gradients()
        cum_x = 0.
        for _ in range(self.n_samples):
            self.first_step()
            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
            diff = (loss_adv - loss)/(self.rho**2)
            cum_x += self.psi(diff)/float(self.n_samples)
            scale = self.psi_p(diff)/float(self.n_samples)
            self.second_step(scale)
        scale = self.phi_p(cum_x)
        self.final_step(scale)
            

@OPTIMIZER_REGISTRY.register()
class FROSAM(torch.optim.Optimizer):
    
    @configurable()
    def __init__(self, params, base_optimizer, rho, n_samples, lam) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.n_samples = n_samples
        self.sample_fn = _sample_gaussian
        self.lam = lam
        super(FROSAM, self).__init__(params, dict(rho=rho))

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
    
    @classmethod
    def from_config(cls, args):
        d = {
            "rho": args.isam_rho,
            "n_samples": args.isam_n_samples,
            "lam": args.isam_lam,
        }
        return d
    
    @torch.no_grad()
    def set_current_gradients(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["g0"] = p.grad
                self.state[p]["cross_sum"] = torch.zeros_like(p.grad, requires_grad=False).to(p)
                self.state[p]["grad_sum"] = torch.zeros_like(p.grad, requires_grad=False).to(p)

    @torch.no_grad()
    def first_step(self):
        for group in self.param_groups:
            valid_ps = [p for p in group["params"] if p.grad is not None]
            for p in valid_ps:
                s = self.sample_fn([tuple(p.grad.shape)], p.device)[0]
                e_w = group["rho"] * s
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self, loss_value):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["cross_sum"] += loss_value*p.grad
                self.state[p]["grad_sum"] += p.grad
                p.sub_(self.state[p]["e_w"])

    @torch.no_grad()
    def final_step(self, loss_sum):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad.set_(
                    self.state[p]["g0"] + self.lam/(self.rho**2)*(
                        1./(self.n_samples-1)*self.state[p]["cross_sum"] - 1./(self.n_samples-1)*loss_sum * 1./self.n_samples*self.state[p]["grad_sum"]
                    )
                )
        self.base_optimizer.step()

    
    @torch.no_grad()
    def step(self, loss, closure=None, **kwargs):
        assert closure is not None, "FROSAM requires closure, which is not provided."

        self.set_current_gradients()
        loss_sum = 0.
        for _ in range(self.n_samples):
            self.first_step()
            self.zero_grad()
            with torch.enable_grad():
                loss_adv = closure()
            loss_sum += loss_adv
            self.second_step(loss_adv)
        self.final_step(loss_sum)



@OPTIMIZER_REGISTRY.register()
class SSAMF(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, rho, sparsity, num_samples, update_freq) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 1.0 <= num_samples, f"num_samples should be greater than 1: {num_samples}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.num_samples = num_samples
        self.update_freq = update_freq
        super(SSAMF, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["num_samples"] = num_samples
            group["update_freq"] = update_freq

        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "num_samples": args.num_samples,
            "update_freq": args.update_freq,
        }
    
    @torch.no_grad()
    def init_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = torch.zeros_like(p, requires_grad=False).to(p)

    @torch.no_grad()
    def update_mask(self, model, train_data, **kwargs):
        fisher_value_dict = {}
        fisher_mask_dict = {}
        for group in self.param_groups:
            for p in group['params']:
                fisher_value_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)
                fisher_mask_dict[id(p)] = torch.zeros_like(p, requires_grad=False).to(p)

        criterion = torch.nn.CrossEntropyLoss()
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=1,
            num_workers=4,
            shuffle=True,
        )
        # cal fisher value
        with torch.enable_grad():
            for idx, (image, label) in enumerate(train_dataloader):
                if idx >= self.num_samples: break
                if idx % (self.num_samples // 10) == 0: print('Updating Mask: [{}/{}]..'.format(idx, self.num_samples))
                image, label = image.cuda(), label.cuda()
                
                output = model(image)
                loss = criterion(output, label)
                loss.backward()

                for group in self.param_groups:
                    for p in group["params"]:
                        fisher_value_dict[id(p)] += torch.square(p.grad).data
                model.zero_grad()
        
        # topk fisher value 
        fisher_value_list = torch.cat([torch.flatten(x) for x in fisher_value_dict.values()])
        
        keep_num = int(len(fisher_value_list) * (1 - self.sparsity))
        _value, _index = torch.topk(fisher_value_list, keep_num)
        
        mask_list = torch.zeros_like(fisher_value_list)
        mask_list.scatter_(0, _index, torch.ones_like(_value))

        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'].to(p)
                self.state[p]['mask'].requires_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, model=None, epoch=None, batch_idx=None, train_data=None, logger=None, **kwargs):
        super().step(closure, **kwargs)
        assert model is not None
        assert train_data is not None
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(model, train_data)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
            
    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num

@OPTIMIZER_REGISTRY.register()
class SSAMD(SAM):
    @configurable()
    def __init__(self, params, base_optimizer, 
        rho, sparsity, drop_rate, drop_strategy, growth_strategy, update_freq, T_start, T_end) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0.0 <= sparsity <= 1.0, f"sparsity should between 0 and 1: {sparsity}"
        assert 0.0 <= drop_rate <= 1.0, f"drop_rate should between 0 and 1: {drop_rate}"
        assert 1.0 <= update_freq , f"update_freq should be greater than 1: {update_freq}"
        self.rho = rho
        self.sparsity = sparsity
        self.drop_rate = drop_rate
        self.drop_strategy = drop_strategy
        self.growth_strategy = growth_strategy
        self.update_freq = update_freq
        self.T_start = T_start
        self.T_end = T_end
        super(SSAMD, self).__init__(params, base_optimizer, rho)

        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["sparsity"] = sparsity
            group["drop_rate"] = drop_rate
            group["drop_strategy"] = drop_strategy
            group["growth_strategy"] = growth_strategy
            group["update_freq"] = update_freq
            group["T_end"] = T_end
        self.init_mask()

    @classmethod
    def from_config(cls, args):
        return {
            "rho": args.rho, 
            "sparsity": args.sparsity,
            "drop_rate": args.drop_rate,
            "drop_strategy": args.drop_strategy,
            "growth_strategy": args.growth_strategy,
            "update_freq": args.update_freq,
            "T_end": args.epochs,
            "T_start": 0,
        }
    
    @torch.no_grad()
    def init_mask(self):
        random_scores = []
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]['score'] = torch.rand(size=p.shape).cpu().data
                random_scores.append(self.state[p]['score'])
        random_scores = torch.cat([torch.flatten(x) for x in random_scores])
        live_num = len(random_scores) - math.ceil(len(random_scores) *self.sparsity)
        _value, _index = torch.topk(random_scores, live_num)

        mask_list = torch.zeros_like(random_scores)
        mask_list.scatter_(0, _index, torch.ones_like(_value))
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['mask'] = mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                del self.state[p]['score']
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
        assert start_index == len(mask_list)
        
    @torch.no_grad()
    def DeathRate_Scheduler(self, epoch):
        dr = (self.drop_rate) * (1 + math.cos(math.pi * (float(epoch - self.T_start) / (self.T_end - self.T_start)))) / 2 
        return dr           

    @torch.no_grad()
    def update_mask(self, epoch, **kwargs):
        death_scores = []
        growth_scores =[]
        for group in self.param_groups:
            for p in group['params']:
                death_score = self.get_score(p, self.drop_strategy)
                death_scores.append((death_score + 1e-7) * self.state[p]['mask'].cpu().data)

                growth_score = self.get_score(p, self.growth_strategy)
                growth_scores.append((growth_score + 1e-7) * (1 - self.state[p]['mask'].cpu().data))
        '''
            Death 
        '''
        death_scores = torch.cat([torch.flatten(x) for x in death_scores])
        death_rate = self.DeathRate_Scheduler(epoch=epoch)
        death_num = int(min((len(death_scores) - len(death_scores) * self.sparsity)* death_rate, len(death_scores) * self.sparsity))
        d_value, d_index = torch.topk(death_scores, int((len(death_scores) - len(death_scores) * self.sparsity) * (1 - death_rate)))

        death_mask_list = torch.zeros_like(death_scores)
        death_mask_list.scatter_(0, d_index, torch.ones_like(d_value))
        '''
            Growth
        '''
        growth_scores = torch.cat([torch.flatten(x) for x in growth_scores])
        growth_num = death_num
        g_value, g_index = torch.topk(growth_scores, growth_num)
        
        growth_mask_list = torch.zeros_like(growth_scores)
        growth_mask_list.scatter_(0, g_index, torch.ones_like(g_value))

        '''
            Mask
        '''
        start_index = 0
        for group in self.param_groups:
            for p in group['params']:
                death_mask = death_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                growth_mask = growth_mask_list[start_index: start_index + p.numel()].reshape(p.shape)
                
                self.state[p]['mask'] = death_mask + growth_mask
                self.state[p]['mask'] = self.state[p]['mask'].to(p)
                self.state[p]['mask'].require_grad = False
                start_index = start_index + p.numel()
                assert self.state[p]['mask'].max() <= 1.0 and self.state[p]['mask'].min() >= 0.0
                
                
                    
        assert start_index == len(death_mask_list)

    def get_score(self, p, score_model=None):
        if score_model == 'weight':
            return torch.abs(p.clone()).cpu().data
        elif score_model == 'gradient':
            return torch.abs(p.grad.clone()).cpu().data
        elif score_model == 'random':
            return torch.rand(size=p.shape).cpu().data
        else:
            raise KeyError    
  
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
                e_w.data = e_w.data * self.state[p]['mask']  # mask the epsilon
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None, epoch=None, batch_idx=None, logger=None, **kwargs):
        assert closure is not None, "SAM requires closure, which is not provided."
        assert epoch is not None
        assert batch_idx is not None
        assert logger is not None

        self.first_step()
        if (epoch % self.update_freq == 0) and (batch_idx == 0):
            logger.log('Update Mask!')
            self.update_mask(epoch)
            logger.log('Mask Lived Weight: {:.4f}'.format(self.mask_info()))
        self.zero_grad()
        with torch.enable_grad():
            closure()
        self.second_step()

    @torch.no_grad()
    def mask_info(self):
        live_num = 0
        total_num = 0
        for group in self.param_groups:
            for p in group['params']:
                live_num += self.state[p]['mask'].sum().item() 
                total_num += self.state[p]['mask'].numel()
        return float(live_num) / total_num
