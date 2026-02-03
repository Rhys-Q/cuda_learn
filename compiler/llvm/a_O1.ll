; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%class.Cat = type { %class.Animal, i32, i8* }
%class.Animal = type { i32 (...)** }

$_ZN3Cat5printEv = comdat any

$_ZTV3Cat = comdat any

$_ZTS3Cat = comdat any

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

$_ZTI3Cat = comdat any

$_ZZN3Cat5printEvE3cnt = comdat any

@.str = private unnamed_addr constant [21 x i8] c"age = %d, name = %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"cat\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"dog\00", align 1
@.str.3 = private unnamed_addr constant [44 x i8] c"\E8\AF\B7\E8\BE\93\E5\85\A5\E5\8A\A8\E7\89\A9\E7\B1\BB\E5\9E\8B\EF\BC\881: \E7\8C\AB, 0: \E7\8B\97\EF\BC\89: \00", align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"\E6\89\93\E5\8D\B0\E7\BB\93\E6\9E\9C: %d\0A\00", align 1
@_ZTV3Cat = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI3Cat to i8*), i8* bitcast (void (%class.Cat*)* @_ZN3Cat5printEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTS3Cat = linkonce_odr dso_local constant [5 x i8] c"3Cat\00", comdat, align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6Animal, i32 0, i32 0) }, comdat, align 8
@_ZTI3Cat = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Cat, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@_ZZN3Cat5printEvE3cnt = linkonce_odr dso_local local_unnamed_addr global i32 0, comdat, align 4
@.str.6 = private unnamed_addr constant [31 x i8] c"age = %d, name = %s, cnt = %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z8PrintDog3Dog(i32 %0, i8* %1) local_unnamed_addr #0 {
  %3 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef %0, i8* noundef %1)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: uwtable
define dso_local noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %0) local_unnamed_addr #2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  br i1 %0, label %2, label %16

2:                                                ; preds = %1
  %3 = call noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #8
  %4 = bitcast i8* %3 to %class.Cat*
  %5 = getelementptr inbounds %class.Cat, %class.Cat* %4, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %5, align 8, !tbaa !5
  %6 = getelementptr inbounds %class.Cat, %class.Cat* %4, i64 0, i32 1
  store i32 1, i32* %6, align 8, !tbaa !8
  %7 = getelementptr inbounds %class.Cat, %class.Cat* %4, i64 0, i32 2
  store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i8** %7, align 8, !tbaa !13
  %8 = bitcast i8* %3 to void (%class.Cat*)***
  br label %9

9:                                                ; preds = %2, %9
  %10 = phi i32 [ 0, %2 ], [ %13, %9 ]
  %11 = load void (%class.Cat*)**, void (%class.Cat*)*** %8, align 8, !tbaa !5
  %12 = load void (%class.Cat*)*, void (%class.Cat*)** %11, align 8
  call void %12(%class.Cat* noundef nonnull align 8 dereferenceable(24) %4)
  %13 = add nuw nsw i32 %10, 1
  %14 = icmp eq i32 %13, 3
  br i1 %14, label %15, label %9, !llvm.loop !14

15:                                               ; preds = %9
  call void @_ZdlPv(i8* noundef %3) #9
  br label %21

16:                                               ; preds = %1, %16
  %17 = phi i32 [ %19, %16 ], [ 0, %1 ]
  %18 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0)) #10
  %19 = add nuw nsw i32 %17, 1
  %20 = icmp eq i32 %19, 3
  br i1 %20, label %21, label %16, !llvm.loop !17

21:                                               ; preds = %16, %15
  %22 = phi i32 [ 1, %15 ], [ 2, %16 ]
  ret i32 %22
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) local_unnamed_addr #4

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) local_unnamed_addr #5

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #6 personality i32 (...)* @__gxx_personality_v0 {
  %1 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %1) #10
  %2 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0))
  %3 = call i32 (i8*, ...) @__isoc99_scanf(i8* noundef getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i8* noundef nonnull %1)
  %4 = load i8, i8* %1, align 1, !tbaa !18, !range !20
  %5 = icmp eq i8 %4, 0
  br i1 %5, label %20, label %6

6:                                                ; preds = %0
  %7 = call noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #8
  %8 = bitcast i8* %7 to %class.Cat*
  %9 = getelementptr inbounds %class.Cat, %class.Cat* %8, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %9, align 8, !tbaa !5
  %10 = getelementptr inbounds %class.Cat, %class.Cat* %8, i64 0, i32 1
  store i32 1, i32* %10, align 8, !tbaa !8
  %11 = getelementptr inbounds %class.Cat, %class.Cat* %8, i64 0, i32 2
  store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0), i8** %11, align 8, !tbaa !13
  %12 = bitcast i8* %7 to void (%class.Cat*)***
  br label %13

13:                                               ; preds = %13, %6
  %14 = phi i32 [ 0, %6 ], [ %17, %13 ]
  %15 = load void (%class.Cat*)**, void (%class.Cat*)*** %12, align 8, !tbaa !5
  %16 = load void (%class.Cat*)*, void (%class.Cat*)** %15, align 8
  call void %16(%class.Cat* noundef nonnull align 8 dereferenceable(24) %8)
  %17 = add nuw nsw i32 %14, 1
  %18 = icmp eq i32 %17, 3
  br i1 %18, label %19, label %13, !llvm.loop !14

19:                                               ; preds = %13
  call void @_ZdlPv(i8* noundef %7) #9
  br label %25

20:                                               ; preds = %0, %20
  %21 = phi i32 [ %23, %20 ], [ 0, %0 ]
  %22 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0)) #10
  %23 = add nuw nsw i32 %21, 1
  %24 = icmp eq i32 %23, 3
  br i1 %24, label %25, label %20, !llvm.loop !17

25:                                               ; preds = %20, %19
  %26 = phi i32 [ 1, %19 ], [ 2, %20 ]
  %27 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), i32 noundef %26)
  %28 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef 2, i8* noundef getelementptr inbounds ([7 x i8], [7 x i8]* @.str.7, i64 0, i64 0)) #10
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %1) #10
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @__isoc99_scanf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN3Cat5printEv(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #7 comdat align 2 {
  %2 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !21
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !21
  %4 = getelementptr inbounds %class.Cat, %class.Cat* %0, i64 0, i32 1
  %5 = load i32, i32* %4, align 8, !tbaa !8
  %6 = getelementptr inbounds %class.Cat, %class.Cat* %0, i64 0, i32 2
  %7 = load i8*, i8** %6, align 8, !tbaa !13
  %8 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([31 x i8], [31 x i8]* @.str.6, i64 0, i64 0), i32 noundef %5, i8* noundef %7, i32 noundef %3)
  ret void
}

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #4 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { norecurse uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { builtin allocsize(0) }
attributes #9 = { builtin nounwind }
attributes #10 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !6, i64 0}
!6 = !{!"vtable pointer", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !10, i64 8}
!9 = !{!"_ZTS3Cat", !10, i64 8, !12, i64 16}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !7, i64 0}
!12 = !{!"any pointer", !11, i64 0}
!13 = !{!9, !12, i64 16}
!14 = distinct !{!14, !15, !16}
!15 = !{!"llvm.loop.mustprogress"}
!16 = !{!"llvm.loop.unroll.disable"}
!17 = distinct !{!17, !15, !16}
!18 = !{!19, !19, i64 0}
!19 = !{!"bool", !11, i64 0}
!20 = !{i8 0, i8 2}
!21 = !{!10, !10, i64 0}
