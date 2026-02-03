; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Dog = type { i32, i8* }
%class.Cat = type { %class.Animal, i32, i8* }
%class.Animal = type { i32 (...)** }

$_ZN3CatC2EiPc = comdat any

$_ZN6AnimalC2Ev = comdat any

$_ZN3Cat5printEv = comdat any

$_ZTV3Cat = comdat any

$_ZTS3Cat = comdat any

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

$_ZTI3Cat = comdat any

$_ZTV6Animal = comdat any

$_ZZN3Cat5printEvE3cnt = comdat any

@.str = private unnamed_addr constant [21 x i8] c"age = %d, name = %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"cat\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"dog\00", align 1
@__const._Z11PrintAnimalb.dog = private unnamed_addr constant %struct.Dog { i32 1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i32 0, i32 0) }, align 8
@.str.3 = private unnamed_addr constant [44 x i8] c"\E8\AF\B7\E8\BE\93\E5\85\A5\E5\8A\A8\E7\89\A9\E7\B1\BB\E5\9E\8B\EF\BC\881: \E7\8C\AB, 0: \E7\8B\97\EF\BC\89: \00", align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"\E6\89\93\E5\8D\B0\E7\BB\93\E6\9E\9C: %d\0A\00", align 1
@_ZL10global_dog = internal global %struct.Dog { i32 2, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.7, i32 0, i32 0) }, align 8
@_ZTV3Cat = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI3Cat to i8*), i8* bitcast (void (%class.Cat*)* @_ZN3Cat5printEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTS3Cat = linkonce_odr dso_local constant [5 x i8] c"3Cat\00", comdat, align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6Animal, i32 0, i32 0) }, comdat, align 8
@_ZTI3Cat = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Cat, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@_ZTV6Animal = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8
@_ZZN3Cat5printEvE3cnt = linkonce_odr dso_local global i32 0, comdat, align 4
@.str.6 = private unnamed_addr constant [31 x i8] c"age = %d, name = %s, cnt = %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @_Z8PrintDog3Dog(i32 %0, i8* %1) #0 {
  %3 = alloca %struct.Dog, align 8
  %4 = bitcast %struct.Dog* %3 to { i32, i8* }*
  %5 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %4, i32 0, i32 0
  store i32 %0, i32* %5, align 8
  %6 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %4, i32 0, i32 1
  store i8* %1, i8** %6, align 8
  %7 = getelementptr inbounds %struct.Dog, %struct.Dog* %3, i32 0, i32 0
  %8 = load i32, i32* %7, align 8
  %9 = getelementptr inbounds %struct.Dog, %struct.Dog* %3, i32 0, i32 1
  %10 = load i8*, i8** %9, align 8
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef %8, i8* noundef %10)
  ret void
}

declare i32 @printf(i8* noundef, ...) #1

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %0) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  %2 = alloca i8, align 1
  %3 = alloca i32, align 4
  %4 = alloca %class.Cat*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %struct.Dog, align 8
  %9 = alloca i32, align 4
  %10 = alloca %struct.Dog, align 8
  %11 = zext i1 %0 to i8
  store i8 %11, i8* %2, align 1
  store i32 0, i32* %3, align 4
  %12 = load i8, i8* %2, align 1
  %13 = trunc i8 %12 to i1
  br i1 %13, label %14, label %40

14:                                               ; preds = %1
  store i32 1, i32* %3, align 4
  %15 = call noalias noundef nonnull i8* @_Znwm(i64 noundef 24) #7
  %16 = bitcast i8* %15 to %class.Cat*
  invoke void @_ZN3CatC2EiPc(%class.Cat* noundef nonnull align 8 dereferenceable(24) %16, i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0))
          to label %17 unwind label %30

17:                                               ; preds = %14
  store %class.Cat* %16, %class.Cat** %4, align 8
  store i32 0, i32* %7, align 4
  br label %18

18:                                               ; preds = %27, %17
  %19 = load i32, i32* %7, align 4
  %20 = icmp slt i32 %19, 3
  br i1 %20, label %21, label %34

21:                                               ; preds = %18
  %22 = load %class.Cat*, %class.Cat** %4, align 8
  %23 = bitcast %class.Cat* %22 to void (%class.Cat*)***
  %24 = load void (%class.Cat*)**, void (%class.Cat*)*** %23, align 8
  %25 = getelementptr inbounds void (%class.Cat*)*, void (%class.Cat*)** %24, i64 0
  %26 = load void (%class.Cat*)*, void (%class.Cat*)** %25, align 8
  call void %26(%class.Cat* noundef nonnull align 8 dereferenceable(24) %22)
  br label %27

27:                                               ; preds = %21
  %28 = load i32, i32* %7, align 4
  %29 = add nsw i32 %28, 1
  store i32 %29, i32* %7, align 4
  br label %18, !llvm.loop !6

30:                                               ; preds = %14
  %31 = landingpad { i8*, i32 }
          cleanup
  %32 = extractvalue { i8*, i32 } %31, 0
  store i8* %32, i8** %5, align 8
  %33 = extractvalue { i8*, i32 } %31, 1
  store i32 %33, i32* %6, align 4
  call void @_ZdlPv(i8* noundef %15) #8
  br label %59

34:                                               ; preds = %18
  %35 = load %class.Cat*, %class.Cat** %4, align 8
  %36 = icmp eq %class.Cat* %35, null
  br i1 %36, label %39, label %37

37:                                               ; preds = %34
  %38 = bitcast %class.Cat* %35 to i8*
  call void @_ZdlPv(i8* noundef %38) #8
  br label %39

39:                                               ; preds = %37, %34
  br label %57

40:                                               ; preds = %1
  store i32 2, i32* %3, align 4
  %41 = bitcast %struct.Dog* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %41, i8* align 8 bitcast (%struct.Dog* @__const._Z11PrintAnimalb.dog to i8*), i64 16, i1 false)
  store i32 0, i32* %9, align 4
  br label %42

42:                                               ; preds = %53, %40
  %43 = load i32, i32* %9, align 4
  %44 = icmp slt i32 %43, 3
  br i1 %44, label %45, label %56

45:                                               ; preds = %42
  %46 = bitcast %struct.Dog* %10 to i8*
  %47 = bitcast %struct.Dog* %8 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %46, i8* align 8 %47, i64 16, i1 false)
  %48 = bitcast %struct.Dog* %10 to { i32, i8* }*
  %49 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %48, i32 0, i32 0
  %50 = load i32, i32* %49, align 8
  %51 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %48, i32 0, i32 1
  %52 = load i8*, i8** %51, align 8
  call void @_Z8PrintDog3Dog(i32 %50, i8* %52)
  br label %53

53:                                               ; preds = %45
  %54 = load i32, i32* %9, align 4
  %55 = add nsw i32 %54, 1
  store i32 %55, i32* %9, align 4
  br label %42, !llvm.loop !8

56:                                               ; preds = %42
  br label %57

57:                                               ; preds = %56, %39
  %58 = load i32, i32* %3, align 4
  ret i32 %58

59:                                               ; preds = %30
  %60 = load i8*, i8** %5, align 8
  %61 = load i32, i32* %6, align 4
  %62 = insertvalue { i8*, i32 } undef, i8* %60, 0
  %63 = insertvalue { i8*, i32 } %62, i32 %61, 1
  resume { i8*, i32 } %63
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) #2

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN3CatC2EiPc(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1, i8* noundef %2) unnamed_addr #3 comdat align 2 {
  %4 = alloca %class.Cat*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8*, align 8
  store %class.Cat* %0, %class.Cat** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8* %2, i8** %6, align 8
  %7 = load %class.Cat*, %class.Cat** %4, align 8
  %8 = bitcast %class.Cat* %7 to %class.Animal*
  call void @_ZN6AnimalC2Ev(%class.Animal* noundef nonnull align 8 dereferenceable(8) %8) #9
  %9 = bitcast %class.Cat* %7 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %9, align 8
  %10 = load i32, i32* %5, align 4
  %11 = getelementptr inbounds %class.Cat, %class.Cat* %7, i32 0, i32 1
  store i32 %10, i32* %11, align 8
  %12 = load i8*, i8** %6, align 8
  %13 = getelementptr inbounds %class.Cat, %class.Cat* %7, i32 0, i32 2
  store i8* %12, i8** %13, align 8
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) #4

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #5

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main() #6 {
  %1 = alloca i32, align 4
  %2 = alloca i8, align 1
  %3 = alloca i32, align 4
  %4 = alloca %struct.Dog, align 8
  store i32 0, i32* %1, align 4
  %5 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0))
  %6 = call i32 (i8*, ...) @__isoc99_scanf(i8* noundef getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i8* noundef %2)
  %7 = load i8, i8* %2, align 1
  %8 = trunc i8 %7 to i1
  %9 = call noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %8)
  store i32 %9, i32* %3, align 4
  %10 = load i32, i32* %3, align 4
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), i32 noundef %10)
  %12 = bitcast %struct.Dog* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %12, i8* align 8 bitcast (%struct.Dog* @_ZL10global_dog to i8*), i64 16, i1 false)
  %13 = bitcast %struct.Dog* %4 to { i32, i8* }*
  %14 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %13, i32 0, i32 0
  %15 = load i32, i32* %14, align 8
  %16 = getelementptr inbounds { i32, i8* }, { i32, i8* }* %13, i32 0, i32 1
  %17 = load i8*, i8** %16, align 8
  call void @_Z8PrintDog3Dog(i32 %15, i8* %17)
  ret i32 0
}

declare i32 @__isoc99_scanf(i8* noundef, ...) #1

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN6AnimalC2Ev(%class.Animal* noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #3 comdat align 2 {
  %2 = alloca %class.Animal*, align 8
  store %class.Animal* %0, %class.Animal** %2, align 8
  %3 = load %class.Animal*, %class.Animal** %2, align 8
  %4 = bitcast %class.Animal* %3 to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV6Animal, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)*** %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline optnone uwtable
define linkonce_odr dso_local void @_ZN3Cat5printEv(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca %class.Cat*, align 8
  store %class.Cat* %0, %class.Cat** %2, align 8
  %3 = load %class.Cat*, %class.Cat** %2, align 8
  %4 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %6 = getelementptr inbounds %class.Cat, %class.Cat* %3, i32 0, i32 1
  %7 = load i32, i32* %6, align 8
  %8 = getelementptr inbounds %class.Cat, %class.Cat* %3, i32 0, i32 2
  %9 = load i8*, i8** %8, align 8
  %10 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4
  %11 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([31 x i8], [31 x i8]* @.str.6, i64 0, i64 0), i32 noundef %7, i8* noundef %9, i32 noundef %10)
  ret void
}

declare void @__cxa_pure_virtual() unnamed_addr

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin allocsize(0) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nobuiltin nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { argmemonly nofree nounwind willreturn }
attributes #6 = { mustprogress noinline norecurse optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { builtin allocsize(0) }
attributes #8 = { builtin nounwind }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
