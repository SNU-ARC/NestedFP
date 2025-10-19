import torch
import cutlass

# 🔧 필수: custom op 네임스페이스 등록
dualfp_lib = torch.library.Library("dualfp", "DEF")


# 글로벌 상태 관리 클래스
class DualFPGlobalState:
    """Global state manager for DualFP quantization mode switching"""
    _use_dualfp = False  # DualFP 사용 여부
    _use_fp8 = True      # DualFP 사용시 fp8/fp16 모드
    
    @classmethod
    def set_dualfp_mode(cls, enable: bool):
        """Enable/disable DualFP quantization"""
        cls._use_dualfp = enable
    
    @classmethod 
    def get_dualfp_mode(cls):
        """Get current DualFP mode"""
        return cls._use_dualfp
    
    @classmethod
    def set_fp8_mode(cls, enable: bool):
        """Set fp8/fp16 mode when DualFP is enabled"""
        cls._use_fp8 = enable
    
    @classmethod 
    def get_fp8_mode(cls):
        """Get current fp8/fp16 mode"""
        return cls._use_fp8
    
    @classmethod
    def set_modes(cls, use_dualfp: bool, use_fp8: bool):
        """Set both modes simultaneously"""
        cls._use_dualfp = use_dualfp
        cls._use_fp8 = use_fp8
    
    @classmethod
    def get_modes(cls):
        """Get both modes as tuple (use_dualfp, use_fp8)"""
        return cls._use_dualfp, cls._use_fp8

@torch.library.custom_op("dualfp::fp16_baseline", mutates_args=({}))
def fp16_baseline(
    M: int,
    N: int,
    K: int,
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    
    D = torch.empty((N, M), dtype=torch.float16, device=A.device)
    

    def select_kernel_fp16_baseline(M: int, N: int, K: int):
        #58 개의 kernel map
        kernel_map = {
            "baseline_1": cutlass.cutlass_tma_warp_specialized_64_16_64,
            "baseline_2": cutlass.cutlass_tma_warp_specialized_64_16_128,
            "baseline_3": cutlass.cutlass_tma_warp_specialized_64_16_256,
            "baseline_4": cutlass.cutlass_tma_warp_specialized_64_32_64,
            "baseline_5": cutlass.cutlass_tma_warp_specialized_64_32_128,
            "baseline_6": cutlass.cutlass_tma_warp_specialized_64_32_256,
            "baseline_7": cutlass.cutlass_tma_warp_specialized_64_64_64,
            "baseline_8": cutlass.cutlass_tma_warp_specialized_64_64_128,
            "baseline_9": cutlass.cutlass_tma_warp_specialized_64_64_256,
            "baseline_10": cutlass.cutlass_tma_warp_specialized_64_128_64,
            "baseline_11": cutlass.cutlass_tma_warp_specialized_64_128_128,
            "baseline_12": cutlass.cutlass_tma_warp_specialized_64_128_256,
            "baseline_13": cutlass.cutlass_tma_warp_specialized_64_256_64,
            "baseline_14": cutlass.cutlass_tma_warp_specialized_64_256_128,
            "baseline_15": cutlass.cutlass_tma_warp_specialized_128_16_64,
            "baseline_16": cutlass.cutlass_tma_warp_specialized_128_16_128,
            "baseline_17": cutlass.cutlass_tma_warp_specialized_128_16_256,
            "baseline_18": cutlass.cutlass_tma_warp_specialized_128_32_64,
            "baseline_19": cutlass.cutlass_tma_warp_specialized_128_32_128,
            "baseline_20": cutlass.cutlass_tma_warp_specialized_128_32_256,
            "baseline_21": cutlass.cutlass_tma_warp_specialized_128_64_64,
            "baseline_22": cutlass.cutlass_tma_warp_specialized_128_64_128,
            "baseline_23": cutlass.cutlass_tma_warp_specialized_128_64_256,
            "baseline_24": cutlass.cutlass_tma_warp_specialized_128_128_64,
            "baseline_25": cutlass.cutlass_tma_warp_specialized_128_128_128,
            "baseline_26": cutlass.cutlass_tma_warp_specialized_128_256_64,
            "baseline_27": cutlass.cutlass_tma_warp_specialized_128_256_128,
            "baseline_28": cutlass.cutlass_tma_warp_specialized_256_16_64,
            "baseline_29": cutlass.cutlass_tma_warp_specialized_256_16_128,
            "baseline_30": cutlass.cutlass_tma_warp_specialized_256_32_64,
            "baseline_31": cutlass.cutlass_tma_warp_specialized_256_32_128,
            "baseline_32": cutlass.cutlass_tma_warp_specialized_256_64_64,
            "baseline_33": cutlass.cutlass_tma_warp_specialized_256_64_128,
            "baseline_34": cutlass.cutlass_tma_warp_specialized_256_128_64,
            "baseline_35": cutlass.cutlass_tma_warp_specialized_256_128_128,
            "baseline_36": cutlass.cutlass_tma_warp_specialized_256_256_64,
            "baseline_37": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_64,
            "baseline_38": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_128,
            "baseline_39": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_16_256,
            "baseline_40": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_64,
            "baseline_41": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_128,
            "baseline_42": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_32_256,
            "baseline_43": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_64,
            "baseline_44": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_128,
            "baseline_45": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_64_256,
            "baseline_46": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_128_64,
            "baseline_47": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_128_128,
            "baseline_48": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_256_64,
            "baseline_49": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_128_256_128,
            "baseline_50": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_16_64,
            "baseline_51": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_16_128,
            "baseline_52": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_32_64,
            "baseline_53": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_32_128,
            "baseline_54": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_64_64,
            "baseline_55": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_64_128,
            "baseline_56": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_128_64,
            "baseline_57": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_128_128,
            "baseline_58": cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_256_256_64,
        }
        
        

        # n, k 에 따라서 optimal kernel 선택.
        if (N, K) == (4096, 4096):
            range_kernel_map = [
                (32, "baseline_3"), (64, "baseline_7"), (96, "baseline_10"),
                (128, "baseline_8"), (160, "baseline_22"), (192, "baseline_21"),
                (224, "baseline_11"), (256, "baseline_10"), (288, "baseline_32"),
                (320, "baseline_24"), (352, "baseline_24"), (384, "baseline_46"),
                (416, "baseline_46"), (448, "baseline_13"), (512, "baseline_13"),
                (704, "baseline_56"), (768, "baseline_48"), (832, "baseline_56"),
                (928, "baseline_48"), (1024, "baseline_48"), (1280, "baseline_13"),
                (1376, "baseline_24"), (1440, "baseline_46"), (1536, "baseline_24"),
                (1600, "baseline_56"), (1664, "baseline_48"), (1760, "baseline_56"),
                (1920, "baseline_48"), (2048, "baseline_48"), (float("inf"), "baseline_48"),
            ]

        elif (N, K) == (6144, 4096):
            range_kernel_map = [
                (32, "baseline_4"), (64, "baseline_7"), (128, "baseline_10"),
                (160, "baseline_46"), (192, "baseline_24"), (256, "baseline_46"),
                (320, "baseline_32"), (416, "baseline_56"), (480, "baseline_48"),
                (640, "baseline_56"), (704, "baseline_54"), (800, "baseline_46"),
                (832, "baseline_48"), (896, "baseline_46"), (928, "baseline_48"),
                (1024, "baseline_46"), (1056, "baseline_56"), (1280, "baseline_48"),
                (1408, "baseline_56"), (1472, "baseline_48"), (1664, "baseline_46"),
                (1952, "baseline_48"), (1984, "baseline_56"), (2048, "baseline_48"),
                (float("inf"), "baseline_48"),
            ]

        elif (N, K) == (28672, 4096):
            range_kernel_map = [(32, "baseline_41"), (64, "baseline_44"), (128, "baseline_57"),
                                (160, "baseline_56"), (256, "baseline_48"), (288, "baseline_56"),
                                (320, "baseline_46"), (416, "baseline_56"), (512, "baseline_48"),
                                (640, "baseline_56"), (768, "baseline_48"), (896, "baseline_56"),
                                (1024, "baseline_48"), (1152, "baseline_56"), (1280, "baseline_48"),
                                (1408, "baseline_56"), (1792, "baseline_48"), (1920, "baseline_56"),
                                (2048, "baseline_48"), (float("inf"), "baseline_48")]

        elif (N, K) == (4096, 14336):
            range_kernel_map = [(32, "baseline_5"), (64, "baseline_7"), (96, "baseline_41"),
                                (128, "baseline_8"), (160, "baseline_44"), (192, "baseline_43"),
                                (224, "baseline_24"), (256, "baseline_10"), (448, "baseline_46"),
                                (512, "baseline_24"), (608, "baseline_48"), (640, "baseline_56"),
                                (768, "baseline_48"), (928, "baseline_56"), (1024, "baseline_48"),
                                (1056, "baseline_24"), (1088, "baseline_46"), (1120, "baseline_24"),
                                (1280, "baseline_46"), (1312, "baseline_48"), (1408, "baseline_46"),
                                (1472, "baseline_48"), (1504, "baseline_13"), (1536, "baseline_46"),
                                (1568, "baseline_56"), (1792, "baseline_48"), (1824, "baseline_56"),
                                (2048, "baseline_48"), (float("inf"), "baseline_48")]

        elif (N, K) == (5120, 4096):
            range_kernel_map = [(32, "baseline_6"), (64, "baseline_7"), (128, "baseline_10"),
                                (256, "baseline_24"), (512, "baseline_48"), (1024, "baseline_24"),
                                (2048, "baseline_48"), (float("inf"), "baseline_48")]
        elif (N, K) == (5120, 32768):
            range_kernel_map = [(32, "baseline_6"), (64, "baseline_8"), (128, "baseline_10"),
                                (256, "baseline_46"), (512, "baseline_48"), (1024, "baseline_24"),
                                (2048, "baseline_48"),(float("inf"), "baseline_48")]
        elif (N, K) == (6144, 5120):
            range_kernel_map = [(32, "baseline_6"), (64, "baseline_8"), (128, "baseline_10"),
                                (256, "baseline_46"), (512, "baseline_56"), (1024, "baseline_46"),
                                (2048, "baseline_48"),(float("inf"), "baseline_48")]
        elif (N, K) == (65536, 5120):
            range_kernel_map = [(32, "baseline_19"), (64, "baseline_44"), (128, "baseline_57"),
                                (256, "baseline_48"), (512, "baseline_48"), (1024, "baseline_48"),
                                (2048, "baseline_48"),(float("inf"), "baseline_48")]
        else:
            range_kernel_map = [(32, "baseline_6"), (64, "baseline_8"), (128, "baseline_10"),
                                (256, "baseline_46"), (512, "baseline_48"), (1024, "baseline_24"),
                                (2048, "baseline_48"),(float("inf"), "baseline_48")]
            
        

        # For debugging purpose. => 
        
        
        # if (N, K) == (4096, 4096):
        #     range_kernel_map = [
        #             (32, "baseline_3"), (64, "baseline_7"), (96, "baseline_10"),
        #             (128, "baseline_8"), (160, "baseline_22"), (192, "baseline_21"),
        #             (224, "baseline_11"), (256, "baseline_10"), (288, "baseline_32"),
        #             (320, "baseline_24"), (352, "baseline_24"), (384, "baseline_46"),
        #             (416, "baseline_46"), (448, "baseline_13"), (512, "baseline_13"),
        #             (704, "baseline_56"), (768, "baseline_48"), (832, "baseline_56"),
        #             (928, "baseline_48"), (1024, "baseline_48"), (1280, "baseline_13"),
        #             (1376, "baseline_24"), (1440, "baseline_46"), (1536, "baseline_24"),
        #             (1600, "baseline_56"), (1664, "baseline_48"), (1760, "baseline_56"),
        #             (1920, "baseline_48"), (2048, "baseline_48"), (float("inf"), "baseline_48"),
        #     ]

        
        # else:
        #     # range_kernel_map = [
        #     #         (32, "baseline_3"), (64, "baseline_7"), (96, "baseline_10"),
        #     #         (128, "baseline_8"), (160, "baseline_22"), (192, "baseline_21"),
        #     #         (224, "baseline_11"), (256, "baseline_10"), (288, "baseline_32"),
        #     #         (320, "baseline_24"), (352, "baseline_24"), (384, "baseline_46"),
        #     #         (416, "baseline_46"), (448, "baseline_13"), (512, "baseline_13"),
        #     #         (704, "baseline_56"), (768, "baseline_48"), (832, "baseline_56"),
        #     #         (928, "baseline_48"), (1024, "baseline_48"), (1280, "baseline_13"),
        #     #         (1376, "baseline_24"), (1440, "baseline_46"), (1536, "baseline_24"),
        #     #         (1600, "baseline_56"), (1664, "baseline_48"), (1760, "baseline_56"),
        #     #         (1920, "baseline_48"), (2048, "baseline_48"), (float("inf"), "baseline_48"),
        #     #   ]
        # range_kernel_map = [
        #         (32, "baseline_3"), (64, "baseline_7"), (96, "baseline_10"),
        #         (128, "baseline_8"), (160, "baseline_22"), (192, "baseline_21"),
        #         (224, "baseline_11"), (256, "baseline_10"), (288, "baseline_32"),
        #         (320, "baseline_24"), (352, "baseline_24"), (384, "baseline_46"),
        #         (416, "baseline_46"), (448, "baseline_13"), (512, "baseline_13"),
        #         (704, "baseline_56"), (768, "baseline_48"), (832, "baseline_56"),
        #         (928, "baseline_48"), (1024, "baseline_48"), (1280, "baseline_13"),
        #         (1376, "baseline_24"), (1440, "baseline_46"), (1536, "baseline_24"),
        #         (1600, "baseline_56"), (1664, "baseline_48"), (1760, "baseline_56"),
        #         (1920, "baseline_48"), (2048, "baseline_48"), (float("inf"), "baseline_48"),
        # ]
        
        
        for upper_bound, kernel_key in range_kernel_map:
            if M <= upper_bound:
                return kernel_map[kernel_key]



        raise ValueError(f"No optimal kernel defined for M={M}")
    
    #The optimal kernel is selected based on the shape of the input tensors.
    kernel = select_kernel_fp16_baseline(M, N, K)
    # Run the kernel with the input tensors and store the result in D
    kernel(A.contiguous(), B.contiguous(), D.contiguous())
    
    return D


@fp16_baseline.register_fake
def _(
    M: int,
    N: int,
    K: int,
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    return torch.empty((N, M), dtype=torch.float16, device=A.device)


@torch.library.custom_op("dualfp::fp16_custom", mutates_args=())
def fp16_custom(
    M: int,
    N: int,
    K: int,
    A1: torch.Tensor,
    A2: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    
    
    # if B.shape[1] != K:
    #     print(f"Error: B.shape[1] ({B.shape[1]}) does not match K ({K})")
    

    def select_kernel_fp16_custom(M: int, N: int, K: int):
        #58 개의 kernel map
        kernel_map = {
            "custom_1": (cutlass.cutlass_tma_warp_specialized_custom_64_16_64, 16),
            "custom_2": (cutlass.cutlass_tma_warp_specialized_custom_64_16_128, 16),
            "custom_3": (cutlass.cutlass_tma_warp_specialized_custom_64_16_256, 16),
            "custom_4": (cutlass.cutlass_tma_warp_specialized_custom_64_32_64, 32),
            "custom_5": (cutlass.cutlass_tma_warp_specialized_custom_64_32_128, 32),
            "custom_6": (cutlass.cutlass_tma_warp_specialized_custom_64_32_256, 32),
            "custom_7": (cutlass.cutlass_tma_warp_specialized_custom_64_64_64, 64),
            "custom_8": (cutlass.cutlass_tma_warp_specialized_custom_64_64_128, 64),
            "custom_9": (cutlass.cutlass_tma_warp_specialized_custom_64_64_256, 64),
            "custom_10": (cutlass.cutlass_tma_warp_specialized_custom_64_128_64, 128),
            "custom_11": (cutlass.cutlass_tma_warp_specialized_custom_64_128_128, 128),
            "custom_12": (cutlass.cutlass_tma_warp_specialized_custom_64_128_256, 128),
            "custom_13": (cutlass.cutlass_tma_warp_specialized_custom_64_256_64,  256),
            "custom_14": (cutlass.cutlass_tma_warp_specialized_custom_64_256_128, 256),
            "custom_15": (cutlass.cutlass_tma_warp_specialized_custom_128_16_64,  16),
            "custom_16": (cutlass.cutlass_tma_warp_specialized_custom_128_16_128, 16),
            "custom_17": (cutlass.cutlass_tma_warp_specialized_custom_128_16_256, 16),
            "custom_18": (cutlass.cutlass_tma_warp_specialized_custom_128_32_64,  32),
            "custom_19": (cutlass.cutlass_tma_warp_specialized_custom_128_32_128, 32),
            "custom_20": (cutlass.cutlass_tma_warp_specialized_custom_128_32_256, 32),
            "custom_21": (cutlass.cutlass_tma_warp_specialized_custom_128_64_64,  64),
            "custom_22": (cutlass.cutlass_tma_warp_specialized_custom_128_64_128, 64),
            "custom_23": (cutlass.cutlass_tma_warp_specialized_custom_128_64_256, 64),
            "custom_24": (cutlass.cutlass_tma_warp_specialized_custom_128_128_64, 128),
            "custom_25": (cutlass.cutlass_tma_warp_specialized_custom_128_128_128,128),
            "custom_26": (cutlass.cutlass_tma_warp_specialized_custom_128_256_64, 256),
            "custom_27": (cutlass.cutlass_tma_warp_specialized_custom_128_256_128,256),
            "custom_28": (cutlass.cutlass_tma_warp_specialized_custom_256_16_64,  16),
            "custom_29": (cutlass.cutlass_tma_warp_specialized_custom_256_16_128, 16),
            "custom_30": (cutlass.cutlass_tma_warp_specialized_custom_256_32_64,  32),
            "custom_31": (cutlass.cutlass_tma_warp_specialized_custom_256_32_128, 32),
            "custom_32": (cutlass.cutlass_tma_warp_specialized_custom_256_64_64,  64),
            "custom_33": (cutlass.cutlass_tma_warp_specialized_custom_256_64_128, 64),
            "custom_34": (cutlass.cutlass_tma_warp_specialized_custom_256_128_64, 128),
            "custom_35": (cutlass.cutlass_tma_warp_specialized_custom_256_128_128,128),
            "custom_36": (cutlass.cutlass_tma_warp_specialized_custom_256_256_64, 256),
            "custom_37": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_16_64,  16),
            "custom_38": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_16_128, 16),
            "custom_39": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_16_256, 16),
            "custom_40": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_32_64,  32),
            "custom_41": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_32_128, 32),
            "custom_42": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_32_256, 32),
            "custom_43": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_64_64,  64),
            "custom_44": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_64_128, 64),
            "custom_45": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_64_256, 64),
            "custom_46": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_128_64,128),
            "custom_47": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_128_128,128),
            "custom_48": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_256_64, 256),
            "custom_49": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_128_256_128,256),
            "custom_50": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_16_64,  16),
            "custom_51": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_16_128, 16),
            "custom_52": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_32_64,  32),
            "custom_53": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_32_128, 32),
            "custom_54": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_64_64,  64),
            "custom_55": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_64_128, 64),
            "custom_56": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_128_64,128),
            "custom_57": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_128_128,128),
            "custom_58": (cutlass.cutlass_tma_warp_specialized_cooperative_2_1_1_custom_256_256_64, 256),
            "stream_1": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_64,  16),
            "stream_2": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_128, 16),
            "stream_3": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_16_256, 16),
            "stream_4": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_64,  32),
            "stream_5": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_128, 32),
            "stream_6": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_32_256, 32),
            "stream_7": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_64,  64),
            "stream_8": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_128, 64),
            "stream_9": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_64_256, 64),
            "stream_10": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_128_64,128),
            "stream_11": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_128_128,128),
            "stream_12": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_256_64, 256),
            "stream_13": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_128_256_128,256),
            "stream_14": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_16_64,  16),
            "stream_15": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_16_128, 16),
            "stream_16": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_32_64,  32),
            "stream_17": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_32_128, 32),
            "stream_18": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_64_64,  64),
            "stream_19": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_64_128, 64),
            "stream_20": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_128_64,128),
            "stream_21": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_128_128,128),
            "stream_22": (cutlass.cutlass_tma_warp_specialized_cooperative_streamk_custom_2_1_1_256_256_64, 256),
        }

        # Llama 3.1 8B
        if (N, K) == (4096, 4096):
           range_kernel_map = [
                (32, "custom_5"),
                (64, "custom_6"),
                (128, "custom_8"),
                (256, "custom_10"),
                (512, "custom_13"),
                (1024, "custom_48"),
                (2048, "custom_48"),
                (float("inf"), "custom_48"),
            ]
        elif (N, K) == (6144, 4096):
           range_kernel_map = [
                (32, "custom_6"),
                (64, "custom_8"),
                (128, "custom_10"),
                (256, "custom_13"),
                (512, "stream_12"),
                (1024, "stream_12"),
                (2048, "custom_48"),
                (float("inf"), "custom_48"),
            ]
        elif (N, K) == (28672, 4096):
          range_kernel_map = [
                (32, "custom_41"),
                (64, "custom_54"),
                (128, "custom_56"),
                (256, "custom_48"),
                (512, "stream_12"),
                (1024, "custom_48"),
                (2048, "custom_48"),
                (float("inf"), "custom_48"),
            ]
        elif (N, K) == (4096, 14336):
            range_kernel_map = [
                    (32, "custom_5"),
                    (64, "custom_8"),
                    (128, "custom_8"),
                    (256, "stream_10"),
                    (512, "stream_12"),
                    (1024, "custom_48"),
                    (2048, "custom_48"),
                    (float("inf"), "custom_48"),
            ]
        #Mistral Small
        elif (N, K) == (5120, 4096):
            # 6, 8, 10, 13, 48, 13, 48
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_8"),
                    (128, "custom_10"),
                    (256, "custom_13"),
                    (512, "stream_12"),
                    (1024, "stream_12"),
                    (2048, "stream_12"),
                    (float("inf"), "custom_48"),
            ]
        elif (N, K) == (5120, 32768):
            # 6, 8, 10, 13, 48, 48, 48
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_8"),
                    (128, "custom_10"),
                    (256, "stream_10"),
                    (512, "stream_12"),
                    (1024, "stream_12"),
                    (2048, "stream_12"),
                    (float("inf"), "custom_48"),
            ]
        elif (N, K) == (6144, 5120):
            # 6, 8, 10, 46, 48, 48, 48
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_8"),
                    (128, "custom_10"),
                    (256, "custom_46"),
                    (512, "custom_48"),
                    (1024, "stream_12"),
                    (2048, "custom_48"),
                    (float("inf"), "custom_48"),
            ]
        elif (N, K) == (65536, 5120):
            # 6, 9, 46, 48, 48, 48, 48
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_9"),
                    (128, "custom_46"),
                    (256, "custom_48"),
                    (512, "custom_48"),
                    (1024, "stream_12"),
                    (2048, "custom_48"),
                    (float("inf"), "custom_48"),
            ]
        #Mistral Nemo-Base
        elif (N, K) == (5120, 14336):
            # 6, 8, 10, 13, 48, 13, 48
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_8"),
                    (128, "custom_10"),
                    (256, "stream_12"),
                    (512, "stream_12"),
                    (1024, "stream_12"),
                    (2048, "stream_12"),
                    (float("inf"), "stream_12"),
            ]
        elif (N, K) == (28672, 5120):
            # 41, 43, 56, 48, 48
            range_kernel_map = [
                    (32, "custom_41"),
                    (64, "custom_43"),
                    (128, "custom_56"),
                    (256, "custom_48"),
                    (512, "stream_12"),
                    (1024, "custom_48"),
                    (2048, "stream_12"),
                    (float("inf"), "stream_12"),
            ]
        # Phi-4
        elif (N,K) == (5120, 5120):
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_8"),
                    (96, "custom_10"),
                    (128, "custom_10"),
                    (160, "custom_43"),
                    (192, "custom_43"),
                    (224, "custom_13"),
                    (256, "custom_13"),
                    (288, "custom_46"),
                    (320, "custom_46"),
                    (352, "custom_46"),
                    (384, "custom_46"),
                    (416, "stream_12"),
                    (448, "stream_12"),
                    (480, "stream_12"),
                    (512, "stream_12"),
                    (float("inf"), "stream_12"),
            ]
        elif (N,K) == (5120, 17920):
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_7"),
                    (96, "stream_10"),
                    (128, "custom_10"),
                    (160, "stream_12"),
                    (192, "stream_10"),
                    (224, "stream_10"),
                    (256, "stream_12"),
                    (288, "stream_20"),
                    (320, "stream_20"),
                    (352, "custom_46"),
                    (384, "stream_20"),
                    (416, "stream_12"),
                    (448, "stream_12"),
                    (480, "stream_12"),
                    (512, "stream_12"),
            ]
        elif (N,K) == (7680, 5120):
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_9"),
                    (96, "custom_11"),
                    (128, "custom_10"),
                    (160, "stream_12"),
                    (192, "stream_12"),
                    (224, "custom_46"),
                    (256, "custom_46"),
                    (288, "custom_56"),
                    (320, "custom_56"),
                    (352, "stream_20"),
                    (384, "custom_56"),
                    (416, "custom_48"),
                    (448, "custom_48"),
                    (480, "custom_48"),
                    (512, "custom_48"),
            ] 
        elif (N,K) == (35840, 5120):
            range_kernel_map = [
                    (32, "custom_6"),
                    (64, "custom_9"),
                    (96, "stream_10"),
                    (128, "stream_10"),
                    (160, "stream_12"),
                    (192, "stream_12"),
                    (224, "stream_12"),
                    (256, "stream_12"),
                    (288, "stream_20"),
                    (320, "stream_20"),
                    (352, "stream_20"),
                    (384, "stream_20"),
                    (416, "stream_12"),
                    (448, "stream_12"),
                    (480, "stream_12"),
                    (512, "stream_12"),
            ]       
        # Base Case
        else:
            range_kernel_map = [
                    (32, "custom_5"),
                    (64, "custom_8"),
                    (128, "custom_8"),
                    (256, "custom_10"),
                    (512, "custom_13"),
                    (1024, "custom_48"),
                    (2048, "custom_48"),
                    (float("inf"), "custom_48"),
            ]

        for upper_bound, kernel_key in range_kernel_map:
            if M <= upper_bound:
                return kernel_map[kernel_key]

    
    #The optimal kernel is selected based on the shape of the input tensors.
    kernel, T2 = select_kernel_fp16_custom(M, N, K)

    #Look Good
    D = torch.empty(N, M, dtype=torch.float16, device=A1.device)
    
    # print("B.shape[1]: ", B.shape[1])
    
    kernel(A1.contiguous(), A2.contiguous(), B.contiguous(), D.contiguous())
    
    D = D.view(M,N)

    return D

        



@fp16_custom.register_fake
def _(
    M: int,
    N: int,
    K: int,
    A1: torch.Tensor,
    A2: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    return torch.empty((M, N), dtype=torch.float16, device=A1.device)






@torch.library.custom_op("dualfp::fp8_custom", mutates_args=())
def fp8_custom(
    M: int,
    N: int,
    K: int,
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    
    
        
    D = torch.empty((N, M), dtype=torch.float16, device=A.device)
    

       
    def select_kernel_fp8_custom(M: int, N: int, K: int):
        # 58 개의 kernel map
        kernel_map = {
            "fp8_1": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_128,
            "fp8_2": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_256,
            "fp8_3": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_16_512,
            "fp8_4": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_128,
            "fp8_5": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_256,
            "fp8_6": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_32_512,
            "fp8_7": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_128,
            "fp8_8": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_256,
            "fp8_9": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_64_512,
            "fp8_10": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_128,
            "fp8_11": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_256,
            "fp8_12": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_128_512,
            "fp8_13": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_256_128,
            "fp8_14": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_64_256_256,
            "fp8_15": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_128,
            "fp8_16": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_256,
            "fp8_17": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_16_512,
            "fp8_18": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_128,
            "fp8_19": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_256,
            "fp8_20": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_32_512,
            "fp8_21": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_128,
            "fp8_22": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_256,
            "fp8_23": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_64_512,
            "fp8_24": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_128_128,
            "fp8_25": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_128_256,
            "fp8_26": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_256_128,
            "fp8_27": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_128_256_256,
            "fp8_28": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_16_128,
            "fp8_29": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_16_256,
            "fp8_30": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_32_128,
            "fp8_31": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_32_256,
            "fp8_32": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_64_128,
            "fp8_33": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_64_256,
            "fp8_34": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_128_128,
            "fp8_35": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_128_256,
            "fp8_36": cutlass.cutlass_tma_warp_specialized_fp8_scale_2_1_1_256_256_128,
            "fp8_37": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_128,
            "fp8_38": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_256,
            "fp8_39": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_16_512,
            "fp8_40": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_128,
            "fp8_41": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_256,
            "fp8_42": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_32_512,
            "fp8_43": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_128,
            "fp8_44": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_256,
            "fp8_45": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_64_512,
            "fp8_46": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_128_128,
            "fp8_47": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_128_256,
            "fp8_48": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_256_128,
            "fp8_49": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_128_256_256,
            "fp8_50": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_16_128,
            "fp8_51": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_16_256,
            "fp8_52": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_32_128,
            "fp8_53": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_32_256,
            "fp8_54": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_64_128,
            "fp8_55": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_64_256,
            "fp8_56": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_128_128,
            "fp8_57": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_128_256,
            "fp8_58": cutlass.cutlass_tma_warp_specialized_cooperative_fp8_scale_2_1_1_256_256_128,
        }
        
        # Llama 3.1 8B
        if (N, K) == (4096, 4096):
            range_kernel_map = [
                (32, "fp8_5"),        # 32~64
                (64, "fp8_5"),       # 65~96
                (128, "fp8_8"),       # 97~128
                (256, "fp8_10"),      # 129~160
                (512, "fp8_24"),      # 161~192
                (1024, "fp8_48"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]

        elif (N, K) == (6144, 4096):
             range_kernel_map = [
                (32, "fp8_6"),        # 32~64
                (64, "fp8_8"),       # 65~96
                (128, "fp8_10"),       # 97~128
                (256, "fp8_46"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_46"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]

        elif (N, K) == (28672, 4096):
             range_kernel_map = [
                (32, "fp8_40"),        # 32~64
                (64, "fp8_43"),       # 65~96
                (128, "fp8_56"),       # 97~128
                (256, "fp8_48"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_48"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]

        elif (N, K) == (4096, 14336):
             range_kernel_map = [
                (32, "fp8_5"),        # 32~64
                (64, "fp8_7"),       # 65~96
                (128, "fp8_8"),       # 97~128
                (256, "fp8_10"),      # 129~160
                (512, "fp8_24"),      # 161~192
                (1024, "fp8_48"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]
        # Mistral Small
        elif (N, K) == (5120, 4096):
            range_kernel_map = [
                (32, "fp8_6"),        # 32~64
                (64, "fp8_8"),       # 65~96
                (128, "fp8_10"),       # 97~128
                (256, "fp8_46"),      # 129~160
                (512, "fp8_56"),      # 161~192
                (1024, "fp8_46"),      # 193~224
                (2048, "fp8_46"),      # 225~256
                (float("inf"), "fp8_46"),
            ]

        elif (N, K) == (5120, 32768):
             range_kernel_map = [
                (32, "fp8_6"),        # 32~64
                (64, "fp8_7"),       # 65~96
                (128, "fp8_10"),       # 97~128
                (256, "fp8_46"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_46"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]

        elif (N, K) == (6144, 5120):
             range_kernel_map = [
                (32, "fp8_6"),        # 32~64
                (64, "fp8_8"),       # 65~96
                (128, "fp8_10"),       # 97~128
                (256, "fp8_46"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_46"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]

        elif (N, K) == (65536, 5120):
             range_kernel_map = [
                (32, "fp8_41"),        # 32~64
                (64, "fp8_44"),       # 65~96
                (128, "fp8_47"),       # 97~128
                (256, "fp8_48"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_48"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]     
             
        # Mistral Nemo
        elif (N, K) == (5120, 14336):
             range_kernel_map = [
                (32, "fp8_5"),        # 32~64
                (64, "fp8_8"),       # 65~96
                (128, "fp8_10"),       # 97~128
                (256, "fp8_46"),      # 129~160
                (512, "fp8_48"),      # 161~192
                (1024, "fp8_46"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]
             
        elif (N, K) == (28672, 5120):
                range_kernel_map = [
                    (32, "fp8_30"),        # 32~64
                    (64, "fp8_44"),       # 65~96
                    (128, "fp8_47"),       # 97~128
                    (256, "fp8_48"),      # 129~160
                    (512, "fp8_48"),      # 161~192
                    (1024, "fp8_48"),      # 193~224
                    (2048, "fp8_48"),      # 225~256
                    (float("inf"), "fp8_48"),
                ]
        
        # Phi-4
        elif (N,K) == (5120, 5120):
            range_kernel_map = [
                    (32, "fp8_5"),
                    (64, "fp8_8"),
                    (96, "fp8_10"),
                    (128, "fp8_46"),
                    (160, "fp8_54"),
                    (192, "fp8_43"),
                    (224, "fp8_46"),
                    (256, "fp8_46"),
                    (288, "fp8_56"),
                    (320, "fp8_56"),
                    (352, "fp8_56"),
                    (384, "fp8_46"),
                    (416, "fp8_48"),
                    (448, "fp8_48"),
                    (480, "fp8_48"),
                    (512, "fp8_48"),
                    (float("inf"), "fp8_48"),
            ]
        elif (N,K) == (5120, 17920):
            range_kernel_map = [
                    (32, "fp8_6"),
                    (64, "fp8_8"),
                    (96, "fp8_43"),
                    (128, "fp8_10"),
                    (160, "fp8_46"),
                    (192, "fp8_44"),
                    (224, "fp8_46"),
                    (256, "fp8_46"),
                    (288, "fp8_24"),
                    (320, "fp8_54"),
                    (352, "fp8_24"),
                    (384, "fp8_24"),
                    (416, "fp8_48"),
                    (448, "fp8_48"),
                    (480, "fp8_48"),
                    (512, "fp8_48"),
            ]
        elif (N,K) == (7680, 5120):
            range_kernel_map = [
                    (32, "fp8_5"),
                    (64, "fp8_8"),
                    (96, "fp8_10"),
                    (128, "fp8_10"),
                    (160, "fp8_47"),
                    (192, "fp8_47"),
                    (224, "fp8_46"),
                    (256, "fp8_24"),
                    (288, "fp8_56"),
                    (320, "fp8_56"),
                    (352, "fp8_56"),
                    (384, "fp8_56"),
                    (416, "fp8_56"),
                    (448, "fp8_48"),
                    (480, "fp8_48"),
                    (512, "fp8_48"),
            ] 
        elif (N,K) == (35840, 5120):
            range_kernel_map = [
                    (32, "fp8_5"),
                    (64, "fp8_8"),
                    (96, "fp8_10"),
                    (128, "fp8_46"),
                    (160, "fp8_54"),
                    (192, "fp8_43"),
                    (224, "fp8_46"),
                    (256, "fp8_46"),
                    (288, "fp8_56"),
                    (320, "fp8_56"),
                    (352, "fp8_56"),
                    (384, "fp8_46"),
                    (416, "fp8_48"),
                    (448, "fp8_48"),
                    (480, "fp8_48"),
                    (512, "fp8_48"),
            ]       
        
        else:
            range_kernel_map = [
                (32, "fp8_5"),        # 32~64
                (64, "fp8_7"),       # 65~96
                (128, "fp8_8"),       # 97~128
                (256, "fp8_10"),      # 129~160
                (512, "fp8_24"),      # 161~192
                (1024, "fp8_48"),      # 193~224
                (2048, "fp8_48"),      # 225~256
                (float("inf"), "fp8_48"),
            ]
        
    

        for upper_bound, kernel_key in range_kernel_map:
            if M <= upper_bound:
                return kernel_map[kernel_key]

        raise ValueError(f"No optimal kernel defined for M={M}")        


    #The optimal kernel is selected based on the shape of the input tensors.
    kernel = select_kernel_fp8_custom(M, N, K)
    
    #safe version for Mistral Small
    # B_safe = torch.empty((M, K), dtype=torch.float16, device=A.device)
    # B_safe[:, :B.shape[1]].copy_(B)
    # # Run the kernel with the input tensors and store the result in D
    # kernel(A, B_safe.contiguous(), D)
    
    
    #else
    kernel(A, B, D)
    
    return D.view(M, N)

    

    # return cutlass.fp8_dualfp(M, N, K, A, B)


@fp8_custom.register_fake
def _(
    M: int,
    N: int,
    K: int,
    A: torch.Tensor,
    B: torch.Tensor
) -> torch.Tensor:
    return torch.empty((M, N), dtype=torch.float16, device=A.device)


@torch.library.custom_op("dualfp::divide_fp16", mutates_args={"D1", "D2"})
def divide_fp16(
    S: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor
) -> None:
    cutlass.divide_fp16(S, D1, D2)


@divide_fp16.register_fake
def _(
    S: torch.Tensor,
    D1: torch.Tensor,
    D2: torch.Tensor
) -> None:
    return None


@torch.library.custom_op("dualfp::merge_fp8", mutates_args={"D"})
def merge_fp8(
    S1: torch.Tensor,
    S2: torch.Tensor,
    D: torch.Tensor
) -> None:
    cutlass.merge_fp8(S1, S2, D)


@merge_fp8.register_fake
def _(
    S1: torch.Tensor,
    S2: torch.Tensor,
    D: torch.Tensor
) -> None:
    return None
