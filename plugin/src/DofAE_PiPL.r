/* DoF Anime - PiPL リソース定義 */
#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "DofAE_Flags.h"

#ifndef AE_OS_WIN
	#include <AE_General.r>
#endif

resource 'PiPL' (16000) {
	{
		Kind {
			AEEffect
		},
		Name {
			DOF_NAME
		},
		Category {
			DOF_CATEGORY
		},
#ifdef AE_OS_WIN
	#ifdef AE_PROC_INTELx64
		CodeWin64X86 {"EffectMain"},
	#endif
#else
	#ifdef AE_OS_MAC
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
	#endif
#endif
		AE_PiPL_Version {
			2,
			0
		},
		AE_Effect_Spec_Version {
			PF_PLUG_IN_VERSION,
			PF_PLUG_IN_SUBVERS
		},
		AE_Effect_Version {
			DOF_PIPL_VERSION
		},
		AE_Effect_Info_Flags {
			0
		},
		AE_Effect_Global_OutFlags {
			DOF_OUTFLAGS
		},
		AE_Effect_Global_OutFlags_2 {
			DOF_OUTFLAGS2
		},
		AE_Effect_Match_Name {
			DOF_MATCH_NAME
		},
		AE_Reserved_Info {
			8
		},
		AE_Effect_Support_URL {
			DOF_SUPPORT_URL
		}
	}
};
