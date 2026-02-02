--- |
  ; ModuleID = 'a.ll'
  source_filename = "a.c"
  target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
  target triple = "x86_64-pc-linux-gnu"
  
  @.str = private unnamed_addr constant [8 x i8] c"c = %d\0A\00", align 1
  
  ; Function Attrs: nofree nounwind uwtable
  define dso_local i32 @main() local_unnamed_addr #0 {
    %1 = tail call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i32 noundef 3)
    ret i32 0
  }
  
  ; Function Attrs: nofree nounwind
  declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1
  
  attributes #0 = { nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
  attributes #1 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
  
  !llvm.module.flags = !{!0, !1, !2, !3}
  !llvm.ident = !{!4}
  
  !0 = !{i32 1, !"wchar_size", i32 4}
  !1 = !{i32 7, !"PIC Level", i32 2}
  !2 = !{i32 7, !"PIE Level", i32 2}
  !3 = !{i32 7, !"uwtable", i32 1}
  !4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}

...
---
name:            main
alignment:       16
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: _, preferred-register: '' }
  - { id: 1, class: _, preferred-register: '' }
  - { id: 2, class: _, preferred-register: '' }
  - { id: 3, class: _, preferred-register: '' }
  - { id: 4, class: _, preferred-register: '' }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  localFrameSize:  0
  savePoint:       ''
  restorePoint:    ''
fixedStack:      []
stack:           []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo: {}
body:             |
  bb.1 (%ir-block.0):
    %2:_(p0) = G_GLOBAL_VALUE @.str
    %1:_(p0) = COPY %2(p0)
    %3:_(s32) = G_CONSTANT i32 3
    %4:_(s32) = G_CONSTANT i32 0
    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def $rsp, implicit-def $eflags, implicit-def $ssp, implicit $rsp, implicit $ssp
    $rdi = COPY %1(p0)
    $esi = COPY %3(s32)
    $al = MOV8ri 0
    CALL64pcrel32 @printf, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $esi, implicit $al, implicit-def $eax
    %0:_(s32) = COPY $eax
    ADJCALLSTACKUP64 0, 0, implicit-def $rsp, implicit-def $eflags, implicit-def $ssp, implicit $rsp, implicit $ssp
    $eax = COPY %4(s32)
    RET 0, implicit $eax

...
